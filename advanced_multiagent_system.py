#!/usr/bin/env python3
"""
Advanced Multi-Agent System with 40+ Agents across 8 Teams
Using Claude Code SDK for sophisticated AI collaboration
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import uuid

from claude_code_sdk import query, ClaudeCodeOptions

# Configuration
MAX_THINKING_TOKENS = 12000


class TeamType(Enum):
    """Different types of teams with specialized capabilities"""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    OPERATIONS = "operations"
    QUALITY = "quality"
    SECURITY = "security"
    STRATEGY = "strategy"


class Priority(Enum):
    """Task priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class Task:
    """Represents a task that can be assigned to agents"""
    id: str
    title: str
    description: str
    priority: Priority
    required_skills: Set[str]
    team_type: Optional[TeamType] = None
    assigned_to: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Message:
    """Inter-agent communication message"""
    id: str
    sender: str
    recipient: str
    content: str
    message_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Base agent class with core capabilities"""
    
    def __init__(self, agent_id: str, name: str, team: str, skills: Set[str],
                 specialization: str, claude_options: ClaudeCodeOptions):
        self.id = agent_id
        self.name = name
        self.team = team
        self.skills = skills
        self.specialization = specialization
        self.claude_options = claude_options
        self.current_task: Optional[Task] = None
        self.task_history: List[Task] = []
        self.message_queue: List[Message] = []
        self.performance_score = 1.0
        self.workload = 0
        self.knowledge_base: Dict[str, Any] = {}
        
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task using Claude"""
        self.current_task = task
        self.workload += 1
        
        prompt = f"""
        You are {self.name}, a {self.specialization} agent in the {self.team} team.
        
        Task: {task.title}
        Description: {task.description}
        Priority: {task.priority.name}
        Required Skills: {', '.join(task.required_skills)}
        
        Your skills: {', '.join(self.skills)}
        
        Please complete this task and provide a structured response.
        """
        
        result = {"agent": self.name, "task_id": task.id, "outputs": []}
        
        try:
            message_count = 0
            async for message in query(prompt=prompt, options=self.claude_options):
                message_count += 1
                
                # Handle different message types
                if hasattr(message, 'content') and message.content:
                    for block in message.content:
                        if hasattr(block, 'text'):
                            result["outputs"].append(block.text)
                        elif hasattr(block, 'name'):  # ToolUseBlock
                            result["outputs"].append({
                                "tool": block.name,
                                "tool_use": True
                            })
                elif hasattr(message, 'subtype'):  # System or Result message
                    if message.subtype == "error":
                        result["error"] = f"System error: {getattr(message, 'data', {})}"
                        break
                
                # Limit messages processed
                if message_count >= 5:
                    break
            
            if "error" not in result:
                task.status = "completed"
                task.completed_at = datetime.now()
                task.result = result
                self.performance_score = min(1.0, self.performance_score + 0.1)
            else:
                task.status = "failed"
                self.performance_score = max(0.1, self.performance_score - 0.1)
            
        except Exception as e:
            result["error"] = f"Exception: {type(e).__name__}: {str(e)}"
            task.status = "failed"
            self.performance_score = max(0.1, self.performance_score - 0.1)
            print(f"Agent {self.name} encountered error: {type(e).__name__}: {e}")
        
        finally:
            self.current_task = None
            self.workload -= 1
            self.task_history.append(task)
            
        return result
    
    async def communicate(self, recipient: 'Agent', content: str, 
                         message_type: str = "info") -> None:
        """Send a message to another agent"""
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.id,
            recipient=recipient.id,
            content=content,
            message_type=message_type
        )
        recipient.receive_message(message)
    
    def receive_message(self, message: Message) -> None:
        """Receive a message from another agent"""
        self.message_queue.append(message)
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent has required skills for a task"""
        return task.required_skills.issubset(self.skills)


class Team:
    """Represents a team of agents with shared goals"""
    
    def __init__(self, team_id: str, name: str, team_type: TeamType):
        self.id = team_id
        self.name = name
        self.team_type = team_type
        self.agents: List[Agent] = []
        self.team_lead: Optional[Agent] = None
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.team_knowledge: Dict[str, Any] = {}
        
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the team"""
        self.agents.append(agent)
        if not self.team_lead and "leadership" in agent.skills:
            self.team_lead = agent
    
    async def assign_task(self, task: Task) -> Optional[Agent]:
        """Assign a task to the most suitable agent"""
        suitable_agents = [
            agent for agent in self.agents 
            if agent.can_handle_task(task) and agent.workload < 3
        ]
        
        if not suitable_agents:
            return None
        
        # Sort by performance and workload
        suitable_agents.sort(
            key=lambda a: (a.performance_score, -a.workload), 
            reverse=True
        )
        
        chosen_agent = suitable_agents[0]
        task.assigned_to = chosen_agent.id
        return chosen_agent
    
    async def collaborate_on_task(self, task: Task) -> Dict[str, Any]:
        """Multiple agents collaborate on a complex task"""
        collaboration_result = {
            "task_id": task.id,
            "team": self.name,
            "agents_involved": [],
            "results": []
        }
        
        # Find agents with required skills
        participating_agents = [
            agent for agent in self.agents
            if any(skill in agent.skills for skill in task.required_skills)
        ][:3]  # Limit to 3 agents for efficiency
        
        if not participating_agents:
            return collaboration_result
        
        # Agents work in parallel
        tasks = []
        for agent in participating_agents:
            collaboration_result["agents_involved"].append(agent.name)
            tasks.append(agent.process_task(task))
        
        results = await asyncio.gather(*tasks)
        collaboration_result["results"] = results
        
        return collaboration_result


class MultiAgentSystem:
    """Main system coordinating all teams and agents"""
    
    def __init__(self):
        self.teams: Dict[str, Team] = {}
        self.agents: Dict[str, Agent] = {}
        self.global_task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.system_knowledge: Dict[str, Any] = {}
        self.inter_team_messages: List[Message] = []
        self._initialize_teams_and_agents()
    
    def _initialize_teams_and_agents(self):
        """Initialize 8 teams with 40+ specialized agents"""
        
        # Define team configurations
        team_configs = {
            TeamType.RESEARCH: {
                "name": "Research & Discovery",
                "agents": [
                    ("Dr. Researcher", {"research", "analysis", "documentation"}, "Senior Research Scientist"),
                    ("Data Hunter", {"data_mining", "web_scraping", "research"}, "Data Research Specialist"),
                    ("Scholar", {"academic_research", "literature_review", "citation"}, "Academic Researcher"),
                    ("Trend Analyst", {"trend_analysis", "market_research", "forecasting"}, "Market Research Analyst"),
                    ("Patent Expert", {"patent_research", "ip_analysis", "legal_research"}, "IP Research Specialist"),
                    ("Science Scout", {"scientific_research", "experiment_design", "hypothesis"}, "Scientific Researcher")
                ]
            },
            TeamType.DEVELOPMENT: {
                "name": "Engineering & Development",
                "agents": [
                    ("Code Master", {"python", "javascript", "architecture", "leadership"}, "Lead Developer"),
                    ("Backend Pro", {"backend", "databases", "api", "python"}, "Backend Engineer"),
                    ("Frontend Wizard", {"frontend", "ui", "react", "design"}, "Frontend Developer"),
                    ("Full Stack Hero", {"fullstack", "devops", "cloud"}, "Full Stack Engineer"),
                    ("Mobile Dev", {"mobile", "ios", "android", "react_native"}, "Mobile Developer"),
                    ("DevOps Ninja", {"devops", "ci_cd", "kubernetes", "automation"}, "DevOps Engineer"),
                    ("ML Engineer", {"machine_learning", "tensorflow", "pytorch", "data_science"}, "ML Engineer")
                ]
            },
            TeamType.ANALYSIS: {
                "name": "Data & Analytics",
                "agents": [
                    ("Data Scientist", {"data_analysis", "statistics", "visualization", "leadership"}, "Lead Data Scientist"),
                    ("Business Analyst", {"business_analysis", "requirements", "process_mapping"}, "Senior Business Analyst"),
                    ("Quant Analyst", {"quantitative_analysis", "financial_modeling", "risk"}, "Quantitative Analyst"),
                    ("Performance Analyst", {"performance_analysis", "metrics", "optimization"}, "Performance Analyst"),
                    ("Market Analyst", {"market_analysis", "competitive_intelligence", "strategy"}, "Market Analyst")
                ]
            },
            TeamType.CREATIVE: {
                "name": "Creative & Design",
                "agents": [
                    ("Creative Director", {"creative_direction", "branding", "leadership", "strategy"}, "Creative Director"),
                    ("UX Designer", {"ux_design", "user_research", "prototyping"}, "Senior UX Designer"),
                    ("UI Artist", {"ui_design", "visual_design", "animation"}, "UI Designer"),
                    ("Content Creator", {"content_creation", "copywriting", "storytelling"}, "Content Strategist"),
                    ("Brand Expert", {"branding", "marketing", "identity_design"}, "Brand Specialist"),
                    ("Video Producer", {"video_production", "editing", "motion_graphics"}, "Video Producer")
                ]
            },
            TeamType.OPERATIONS: {
                "name": "Operations & Management",
                "agents": [
                    ("Ops Manager", {"operations", "management", "leadership", "planning"}, "Operations Manager"),
                    ("Project Lead", {"project_management", "agile", "scrum", "coordination"}, "Project Manager"),
                    ("Process Expert", {"process_improvement", "six_sigma", "efficiency"}, "Process Engineer"),
                    ("Resource Planner", {"resource_planning", "capacity_management", "scheduling"}, "Resource Manager"),
                    ("Supply Chain Pro", {"supply_chain", "logistics", "inventory"}, "Supply Chain Manager")
                ]
            },
            TeamType.QUALITY: {
                "name": "Quality & Testing",
                "agents": [
                    ("QA Lead", {"quality_assurance", "test_strategy", "leadership"}, "QA Lead"),
                    ("Test Automation", {"test_automation", "selenium", "pytest", "ci"}, "Automation Engineer"),
                    ("Manual Tester", {"manual_testing", "test_cases", "bug_tracking"}, "QA Analyst"),
                    ("Performance Tester", {"performance_testing", "load_testing", "jmeter"}, "Performance Test Engineer"),
                    ("Security Tester", {"security_testing", "penetration_testing", "vulnerability"}, "Security Test Engineer")
                ]
            },
            TeamType.SECURITY: {
                "name": "Security & Compliance",
                "agents": [
                    ("Security Chief", {"security_architecture", "leadership", "risk_management"}, "Chief Security Officer"),
                    ("Ethical Hacker", {"penetration_testing", "ethical_hacking", "vulnerability_assessment"}, "Senior Penetration Tester"),
                    ("Compliance Officer", {"compliance", "regulations", "audit", "gdpr"}, "Compliance Manager"),
                    ("Incident Handler", {"incident_response", "forensics", "threat_hunting"}, "Incident Response Lead"),
                    ("Crypto Expert", {"cryptography", "blockchain", "secure_coding"}, "Cryptography Specialist")
                ]
            },
            TeamType.STRATEGY: {
                "name": "Strategy & Innovation",
                "agents": [
                    ("Strategy Chief", {"strategic_planning", "leadership", "vision", "innovation"}, "Chief Strategy Officer"),
                    ("Innovation Lead", {"innovation", "r&d", "emerging_tech", "ideation"}, "Innovation Manager"),
                    ("Business Strategist", {"business_strategy", "competitive_analysis", "growth"}, "Senior Strategist"),
                    ("Digital Transformer", {"digital_transformation", "change_management", "modernization"}, "Digital Transformation Lead"),
                    ("Future Thinker", {"futurism", "trend_forecasting", "scenario_planning"}, "Futurist")
                ]
            }
        }
        
        # Create teams and agents
        for team_type, config in team_configs.items():
            team_id = f"team_{team_type.value}"
            team = Team(team_id, config["name"], team_type)
            
            for i, (name, skills, specialization) in enumerate(config["agents"]):
                agent_id = f"{team_type.value}_{i+1}"
                
                # Configure Claude options based on agent specialization
                tools = self._get_tools_for_specialization(specialization)
                claude_options = ClaudeCodeOptions(
                    allowed_tools=tools,
                    max_thinking_tokens=MAX_THINKING_TOKENS,
                    permission_mode="bypassPermissions",
                    model="claude-3-haiku-20240307",  # Use fast model for agents
                    max_turns=1  # Limit to single turn per task
                )
                
                agent = Agent(
                    agent_id=agent_id,
                    name=name,
                    team=team.name,
                    skills=skills,
                    specialization=specialization,
                    claude_options=claude_options
                )
                
                team.add_agent(agent)
                self.agents[agent_id] = agent
            
            self.teams[team_id] = team
    
    def _get_tools_for_specialization(self, specialization: str) -> List[str]:
        """Get appropriate tools based on agent specialization"""
        base_tools = ["Read", "Edit", "Write"]
        
        if "Developer" in specialization or "Engineer" in specialization:
            return base_tools + ["Bash", "Grep", "MultiEdit"]
        elif "Analyst" in specialization or "Researcher" in specialization:
            return base_tools + ["WebSearch", "Grep"]
        elif "Designer" in specialization or "Creative" in specialization:
            return base_tools + ["WebFetch"]
        elif "Security" in specialization:
            return base_tools + ["Bash", "Grep"]
        else:
            return base_tools
    
    async def route_task(self, task: Task) -> Dict[str, Any]:
        """Route a task to the appropriate team and agent"""
        # Determine best team based on task requirements
        if task.team_type:
            team_id = f"team_{task.team_type.value}"
            team = self.teams.get(team_id)
        else:
            # Find team with most matching skills
            best_team = None
            best_score = 0
            
            for team in self.teams.values():
                score = sum(
                    1 for agent in team.agents
                    if agent.can_handle_task(task)
                )
                if score > best_score:
                    best_score = score
                    best_team = team
            
            team = best_team
        
        if not team:
            return {"error": "No suitable team found"}
        
        # Complex tasks require collaboration
        if task.priority == Priority.CRITICAL or len(task.required_skills) > 3:
            return await team.collaborate_on_task(task)
        
        # Simple tasks go to individual agents
        agent = await team.assign_task(task)
        if agent:
            return await agent.process_task(task)
        
        return {"error": "No available agent"}
    
    async def execute_workflow(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """Execute a workflow with multiple tasks"""
        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort(tasks)
        
        results = []
        for task in sorted_tasks:
            # Check if dependencies are completed
            if all(
                dep in [t.id for t in self.completed_tasks]
                for dep in task.dependencies
            ):
                result = await self.route_task(task)
                results.append(result)
                
                if task.status == "completed":
                    self.completed_tasks.append(task)
        
        return results
    
    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks based on dependencies"""
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for task in tasks:
            for dep in task.dependencies:
                graph[dep].append(task.id)
                in_degree[task.id] += 1
        
        # Find tasks with no dependencies
        queue = [task for task in tasks if in_degree[task.id] == 0]
        sorted_tasks = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda t: t.priority.value, reverse=True)
            task = queue.pop(0)
            sorted_tasks.append(task)
            
            # Update dependencies
            for dependent_id in graph[task.id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    dependent_task = next(t for t in tasks if t.id == dependent_id)
                    queue.append(dependent_task)
        
        return sorted_tasks
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        status = {
            "teams": {},
            "total_agents": len(self.agents),
            "active_tasks": 0,
            "completed_tasks": len(self.completed_tasks),
            "system_performance": 0
        }
        
        for team_id, team in self.teams.items():
            team_status = {
                "name": team.name,
                "type": team.team_type.value,
                "agents": len(team.agents),
                "active_agents": sum(1 for a in team.agents if a.current_task),
                "completed_tasks": len(team.completed_tasks),
                "average_performance": sum(a.performance_score for a in team.agents) / len(team.agents)
            }
            status["teams"][team_id] = team_status
            status["active_tasks"] += team_status["active_agents"]
        
        # Calculate overall system performance
        all_performances = [a.performance_score for a in self.agents.values()]
        status["system_performance"] = sum(all_performances) / len(all_performances)
        
        return status
    
    async def inter_team_collaboration(self, task_description: str, 
                                     teams_involved: List[TeamType]) -> Dict[str, Any]:
        """Facilitate collaboration between multiple teams"""
        collaboration_id = str(uuid.uuid4())
        results = {
            "collaboration_id": collaboration_id,
            "teams": [],
            "messages": [],
            "outcomes": []
        }
        
        # Create a complex task requiring multiple teams
        main_task = Task(
            id=f"collab_{collaboration_id}",
            title=f"Inter-team Collaboration: {task_description}",
            description=task_description,
            priority=Priority.HIGH,
            required_skills=set()
        )
        
        # Each team contributes their expertise
        for team_type in teams_involved:
            team = self.teams[f"team_{team_type.value}"]
            results["teams"].append(team.name)
            
            # Select representative agents
            if team.team_lead:
                lead = team.team_lead
            else:
                lead = team.agents[0]
            
            # Process task from team perspective
            team_result = await lead.process_task(main_task)
            results["outcomes"].append(team_result)
            
            # Share results with other teams
            for other_team_type in teams_involved:
                if other_team_type != team_type:
                    other_team = self.teams[f"team_{other_team_type.value}"]
                    if other_team.team_lead:
                        await lead.communicate(
                            other_team.team_lead,
                            f"Team {team.name} findings: {json.dumps(team_result)}",
                            "collaboration"
                        )
        
        return results


async def demonstrate_system():
    """Demonstrate the multi-agent system capabilities"""
    print("🚀 Initializing Advanced Multi-Agent System with 40+ Agents across 8 Teams")
    print("=" * 80)
    
    system = MultiAgentSystem()
    
    # Display system overview
    status = system.get_system_status()
    print(f"\n📊 System Overview:")
    print(f"Total Agents: {status['total_agents']}")
    print(f"Teams: {len(status['teams'])}")
    
    for team_id, team_info in status["teams"].items():
        print(f"\n  🏢 {team_info['name']} ({team_info['type']})")
        print(f"     Agents: {team_info['agents']}")
        print(f"     Average Performance: {team_info['average_performance']:.2f}")
    
    # Example 1: Simple task routing
    print("\n\n📋 Example 1: Simple Task Routing")
    print("-" * 40)
    
    simple_task = Task(
        id="task_001",
        title="Analyze website performance",
        description="Analyze the performance of our website and provide optimization recommendations",
        priority=Priority.MEDIUM,
        required_skills={"performance_analysis", "web_development"},
        team_type=TeamType.ANALYSIS
    )
    
    result1 = await system.route_task(simple_task)
    print(f"Task assigned to: {result1.get('agent', 'Unknown')}")
    print(f"Status: {'Completed' if simple_task.status == 'completed' else 'Failed'}")
    
    # Example 2: Complex workflow with dependencies
    print("\n\n📋 Example 2: Complex Workflow with Dependencies")
    print("-" * 40)
    
    workflow_tasks = [
        Task(
            id="research_001",
            title="Research AI trends",
            description="Research current AI trends and emerging technologies",
            priority=Priority.HIGH,
            required_skills={"research", "trend_analysis"},
            team_type=TeamType.RESEARCH
        ),
        Task(
            id="design_001",
            title="Design AI product concept",
            description="Based on research, design an innovative AI product concept",
            priority=Priority.HIGH,
            required_skills={"creative_direction", "product_design"},
            team_type=TeamType.CREATIVE,
            dependencies=["research_001"]
        ),
        Task(
            id="develop_001",
            title="Create MVP prototype",
            description="Develop a minimum viable prototype of the AI product",
            priority=Priority.CRITICAL,
            required_skills={"python", "machine_learning", "fullstack"},
            team_type=TeamType.DEVELOPMENT,
            dependencies=["design_001"]
        ),
        Task(
            id="test_001",
            title="Test and validate prototype",
            description="Comprehensive testing of the prototype",
            priority=Priority.HIGH,
            required_skills={"test_automation", "quality_assurance"},
            team_type=TeamType.QUALITY,
            dependencies=["develop_001"]
        )
    ]
    
    workflow_results = await system.execute_workflow(workflow_tasks)
    print(f"Workflow executed: {len(workflow_results)} tasks completed")
    
    # Example 3: Inter-team collaboration
    print("\n\n📋 Example 3: Inter-Team Collaboration")
    print("-" * 40)
    
    collab_result = await system.inter_team_collaboration(
        "Design and implement a secure, high-performance e-commerce platform",
        [TeamType.STRATEGY, TeamType.DEVELOPMENT, TeamType.SECURITY, TeamType.CREATIVE]
    )
    
    print(f"Collaboration ID: {collab_result['collaboration_id']}")
    print(f"Teams involved: {', '.join(collab_result['teams'])}")
    print(f"Outcomes generated: {len(collab_result['outcomes'])}")
    
    # Example 4: Critical incident response
    print("\n\n🚨 Example 4: Critical Incident Response")
    print("-" * 40)
    
    incident_task = Task(
        id="incident_001",
        title="Critical Security Breach Response",
        description="Respond to detected security breach, assess damage, and implement fixes",
        priority=Priority.CRITICAL,
        required_skills={"incident_response", "security_architecture", "forensics", 
                        "devops", "communication"},
        team_type=TeamType.SECURITY
    )
    
    incident_result = await system.route_task(incident_task)
    print(f"Incident response team activated")
    print(f"Agents involved: {len(incident_result.get('agents_involved', []))}")
    
    # Final system status
    print("\n\n📊 Final System Status")
    print("=" * 80)
    
    final_status = system.get_system_status()
    print(f"Total completed tasks: {final_status['completed_tasks']}")
    print(f"Active tasks: {final_status['active_tasks']}")
    print(f"System performance: {final_status['system_performance']:.2f}")
    
    # Performance breakdown by team
    print("\n🏆 Team Performance Rankings:")
    team_rankings = sorted(
        final_status["teams"].items(),
        key=lambda x: x[1]["average_performance"],
        reverse=True
    )
    
    for i, (team_id, team_info) in enumerate(team_rankings, 1):
        print(f"{i}. {team_info['name']}: {team_info['average_performance']:.2f}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_system())