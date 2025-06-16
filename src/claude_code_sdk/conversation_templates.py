"""Conversation templates and presets for common workflows."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from .types import ClaudeCodeOptions


class TemplateType(Enum):
    """Types of conversation templates."""
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    LEARNING = "learning"
    PAIR_PROGRAMMING = "pair_programming"
    ARCHITECTURE = "architecture"
    CUSTOM = "custom"


@dataclass
class ConversationTemplate:
    """Template for structured conversations."""
    name: str
    type: TemplateType
    description: str
    system_prompt: Optional[str] = None
    initial_prompts: List[str] = field(default_factory=list)
    required_context: List[str] = field(default_factory=list)  # e.g., ["file_path", "language"]
    options_preset: Optional[ClaudeCodeOptions] = None
    follow_up_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def create_options(self, **overrides) -> ClaudeCodeOptions:
        """Create options with template presets and overrides."""
        if self.options_preset:
            # Create a copy of the preset
            options = ClaudeCodeOptions(
                allowed_tools=self.options_preset.allowed_tools.copy(),
                max_thinking_tokens=self.options_preset.max_thinking_tokens,
                system_prompt=self.options_preset.system_prompt,
                append_system_prompt=self.options_preset.append_system_prompt,
                permission_mode=self.options_preset.permission_mode,
                max_turns=self.options_preset.max_turns,
            )
        else:
            options = ClaudeCodeOptions()
        
        # Apply template system prompt
        if self.system_prompt:
            options.system_prompt = self.system_prompt
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(options, key):
                setattr(options, key, value)
        
        return options
    
    def format_initial_prompt(self, context: Dict[str, str]) -> str:
        """Format the initial prompt with provided context."""
        # Check required context
        missing = [key for key in self.required_context if key not in context]
        if missing:
            raise ValueError(f"Missing required context: {missing}")
        
        # Format the first prompt
        if self.initial_prompts:
            return self.initial_prompts[0].format(**context)
        return ""


# Pre-defined templates
CODE_REVIEW_TEMPLATE = ConversationTemplate(
    name="Code Review Assistant",
    type=TemplateType.CODE_REVIEW,
    description="Comprehensive code review with best practices and suggestions",
    system_prompt="""You are an expert code reviewer. Focus on:
- Code quality and readability
- Performance implications
- Security concerns
- Best practices for the language
- Potential bugs or edge cases
Be constructive and provide specific suggestions.""",
    initial_prompts=[
        "Please review the code in {file_path} and provide detailed feedback on code quality, potential issues, and improvement suggestions.",
    ],
    required_context=["file_path"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Read", "Grep", "Edit"],
        permission_mode="default",
    ),
    follow_up_suggestions=[
        "Can you explain the security implications?",
        "What are the performance characteristics?",
        "How would you refactor this code?",
        "Are there any design pattern improvements?",
    ],
)

DEBUGGING_TEMPLATE = ConversationTemplate(
    name="Interactive Debugger",
    type=TemplateType.DEBUGGING,
    description="Step-by-step debugging assistance",
    system_prompt="""You are a debugging expert. Approach problems systematically:
1. Understand the expected vs actual behavior
2. Identify potential causes
3. Use tools to investigate
4. Test hypotheses
5. Provide clear fixes
Always explain your reasoning.""",
    initial_prompts=[
        "I'm experiencing {issue_description}. The error occurs in {file_path}. Can you help me debug this?",
    ],
    required_context=["issue_description", "file_path"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Read", "Edit", "Bash", "Grep"],
        permission_mode="acceptEdits",
    ),
    follow_up_suggestions=[
        "Can you add logging to trace the issue?",
        "What test would catch this bug?",
        "Are there similar issues elsewhere?",
        "How can we prevent this in the future?",
    ],
)

REFACTORING_TEMPLATE = ConversationTemplate(
    name="Refactoring Assistant",
    type=TemplateType.REFACTORING,
    description="Systematic code refactoring with preservation of functionality",
    system_prompt="""You are a refactoring expert. When refactoring:
1. Preserve all existing functionality
2. Improve code structure and readability
3. Apply design patterns where appropriate
4. Ensure backward compatibility
5. Write clear commit messages
Explain each refactoring decision.""",
    initial_prompts=[
        "Please refactor {file_path} to improve {improvement_goal}. Maintain all existing functionality.",
    ],
    required_context=["file_path", "improvement_goal"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Read", "Edit", "Write", "Bash"],
        permission_mode="acceptEdits",
    ),
    follow_up_suggestions=[
        "Can you add unit tests for the refactored code?",
        "What other files might need similar refactoring?",
        "Can you create a migration guide?",
        "Are there performance implications?",
    ],
)

TESTING_TEMPLATE = ConversationTemplate(
    name="Test Writer",
    type=TemplateType.TESTING,
    description="Comprehensive test generation and improvement",
    system_prompt="""You are a testing expert. When writing tests:
1. Aim for high code coverage
2. Test edge cases and error conditions
3. Use appropriate testing patterns
4. Make tests readable and maintainable
5. Include both positive and negative test cases
Follow the project's testing conventions.""",
    initial_prompts=[
        "Please write comprehensive tests for {file_path} using {test_framework}.",
    ],
    required_context=["file_path", "test_framework"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        permission_mode="acceptEdits",
    ),
    follow_up_suggestions=[
        "Can you add edge case tests?",
        "What about integration tests?",
        "Can you improve test coverage?",
        "Should we add property-based tests?",
    ],
)

DOCUMENTATION_TEMPLATE = ConversationTemplate(
    name="Documentation Writer",
    type=TemplateType.DOCUMENTATION,
    description="Create and improve documentation",
    system_prompt="""You are a technical documentation expert. Focus on:
1. Clear and concise explanations
2. Practical examples
3. API documentation
4. Usage patterns
5. Troubleshooting guides
Match the project's documentation style.""",
    initial_prompts=[
        "Please create comprehensive documentation for {component} in {file_path}.",
    ],
    required_context=["component", "file_path"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Read", "Write", "Edit"],
        permission_mode="acceptEdits",
    ),
    follow_up_suggestions=[
        "Can you add more examples?",
        "What about error handling documentation?",
        "Can you create a quick start guide?",
        "Should we add API reference docs?",
    ],
)

PAIR_PROGRAMMING_TEMPLATE = ConversationTemplate(
    name="Pair Programming Partner",
    type=TemplateType.PAIR_PROGRAMMING,
    description="Collaborative coding session",
    system_prompt="""You are a pair programming partner. Be collaborative:
1. Suggest improvements as we code
2. Catch potential issues early
3. Propose alternative approaches
4. Help with naming and structure
5. Keep the session productive
Think out loud and explain your reasoning.""",
    initial_prompts=[
        "Let's work on {task_description}. I'll start with {starting_point}.",
    ],
    required_context=["task_description", "starting_point"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Grep"],
        permission_mode="acceptEdits",
        max_turns=50,  # Longer session expected
    ),
    follow_up_suggestions=[
        "What should we implement next?",
        "Is there a better approach?",
        "Should we refactor this part?",
        "Can you write a test for this?",
    ],
)

ARCHITECTURE_TEMPLATE = ConversationTemplate(
    name="Architecture Advisor",
    type=TemplateType.ARCHITECTURE,
    description="System design and architecture discussions",
    system_prompt="""You are a software architect. Consider:
1. System scalability and performance
2. Maintainability and modularity
3. Design patterns and principles
4. Technology choices and trade-offs
5. Future extensibility
Provide diagrams and concrete examples where helpful.""",
    initial_prompts=[
        "I need to design {system_description}. The main requirements are {requirements}.",
    ],
    required_context=["system_description", "requirements"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Write", "Edit"],
        permission_mode="acceptEdits",
    ),
    follow_up_suggestions=[
        "How would this scale to more users?",
        "What are the security considerations?",
        "Can you create a sequence diagram?",
        "What technologies would you recommend?",
    ],
)

LEARNING_TEMPLATE = ConversationTemplate(
    name="Learning Assistant",
    type=TemplateType.LEARNING,
    description="Educational conversations about code and concepts",
    system_prompt="""You are a patient teacher. When explaining:
1. Start with fundamentals
2. Use clear examples
3. Build up complexity gradually
4. Check understanding
5. Provide exercises
Adapt to the learner's level.""",
    initial_prompts=[
        "Can you explain {concept} and how it's used in {context}?",
    ],
    required_context=["concept", "context"],
    options_preset=ClaudeCodeOptions(
        allowed_tools=["Write", "Read"],
        permission_mode="acceptEdits",
    ),
    follow_up_suggestions=[
        "Can you show a practical example?",
        "What are common mistakes to avoid?",
        "How does this compare to alternatives?",
        "Can you suggest exercises to practice?",
    ],
)


# Template registry
TEMPLATES: Dict[str, ConversationTemplate] = {
    "code_review": CODE_REVIEW_TEMPLATE,
    "debugging": DEBUGGING_TEMPLATE,
    "refactoring": REFACTORING_TEMPLATE,
    "testing": TESTING_TEMPLATE,
    "documentation": DOCUMENTATION_TEMPLATE,
    "pair_programming": PAIR_PROGRAMMING_TEMPLATE,
    "architecture": ARCHITECTURE_TEMPLATE,
    "learning": LEARNING_TEMPLATE,
}


class TemplateManager:
    """Manages conversation templates."""
    
    def __init__(self):
        self.templates = TEMPLATES.copy()
        self.custom_templates: Dict[str, ConversationTemplate] = {}
    
    def get_template(self, name: str) -> Optional[ConversationTemplate]:
        """Get a template by name."""
        return self.templates.get(name) or self.custom_templates.get(name)
    
    def list_templates(self, type: Optional[TemplateType] = None) -> List[ConversationTemplate]:
        """List available templates."""
        all_templates = list(self.templates.values()) + list(self.custom_templates.values())
        
        if type:
            return [t for t in all_templates if t.type == type]
        return all_templates
    
    def create_custom_template(
        self,
        name: str,
        description: str,
        system_prompt: str,
        initial_prompts: List[str],
        required_context: List[str],
        **kwargs
    ) -> ConversationTemplate:
        """Create a custom template."""
        template = ConversationTemplate(
            name=name,
            type=TemplateType.CUSTOM,
            description=description,
            system_prompt=system_prompt,
            initial_prompts=initial_prompts,
            required_context=required_context,
            **kwargs
        )
        self.custom_templates[name] = template
        return template
    
    def suggest_template(self, task_description: str) -> Optional[ConversationTemplate]:
        """Suggest a template based on task description."""
        # Simple keyword matching (could be enhanced with better NLP)
        keywords = {
            "review": "code_review",
            "debug": "debugging",
            "error": "debugging",
            "refactor": "refactoring",
            "clean": "refactoring",
            "test": "testing",
            "document": "documentation",
            "docs": "documentation",
            "pair": "pair_programming",
            "design": "architecture",
            "learn": "learning",
            "explain": "learning",
        }
        
        task_lower = task_description.lower()
        for keyword, template_name in keywords.items():
            if keyword in task_lower:
                return self.templates.get(template_name)
        
        return None