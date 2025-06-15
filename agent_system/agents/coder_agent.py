"""
Coder Agent - Specializes in code generation and tool creation.
"""

import asyncio
import ast
import json
from typing import Dict, Any, List, Optional
import textwrap

from ..core.base_agent import BaseAgent
from ..core.task import Task, TaskResult, TaskType
from ..config import AgentRole, AgentCapability
from ..integrations import get_tool_registry_client, ToolBuilder


class CoderAgent(BaseAgent):
    """Agent specialized in code generation and tool creation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.CODER
        self.capabilities = {
            AgentCapability.CODE_GENERATION,
            AgentCapability.TOOL_CREATION,
            AgentCapability.TOOL_EXECUTION
        }
        self.tool_registry = get_tool_registry_client()
        
        # Code generation templates and patterns
        self.code_templates = self._load_code_templates()
        self.created_tools: List[Dict[str, Any]] = []
    
    async def initialize(self):
        """Initialize the coder agent."""
        self.logger.info("Initializing Coder Agent")
        
        # Verify tool registry connection
        try:
            health = await self.tool_registry.health_check()
            self.logger.info(f"Tool registry connected: {health['status']}")
        except Exception as e:
            self.logger.error(f"Tool registry connection failed: {e}")
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process coding and tool creation tasks."""
        self.logger.info(f"Processing coding task: {task.name}")
        
        try:
            if task.type == TaskType.TOOL_CREATION:
                return await self._create_tool(task)
            elif task.type == TaskType.CODE_GENERATION:
                return await self._generate_code(task)
            else:
                return await self._handle_general_coding_task(task)
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _create_tool(self, task: Task) -> TaskResult:
        """Create a new tool based on specifications."""
        tool_spec = task.input_data
        
        # Extract tool requirements
        name = tool_spec.get("name", f"tool_{task.id[:8]}")
        description = tool_spec.get("description", "Auto-generated tool")
        tool_type = tool_spec.get("type", "python")  # python or http
        functionality = tool_spec.get("functionality", {})
        
        self.logger.info(f"Creating {tool_type} tool: {name}")
        
        try:
            if tool_type == "python":
                tool_data = await self._create_python_tool(
                    name, description, functionality
                )
            elif tool_type == "http":
                tool_data = await self._create_http_tool(
                    name, description, functionality
                )
            else:
                raise ValueError(f"Unknown tool type: {tool_type}")
            
            # Register the tool
            created_tool = await self.tool_registry.create_tool(
                tool_data, 
                agent_id=self.id
            )
            
            self.created_tools.append(created_tool)
            self.metrics.tools_created += 1
            
            return TaskResult(
                success=True,
                output=created_tool,
                metadata={
                    "tool_id": created_tool["id"],
                    "tool_name": created_tool["name"],
                    "validation_status": "pending"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Tool creation failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _create_python_tool(
        self, 
        name: str, 
        description: str, 
        functionality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a Python-based tool."""
        # Generate function code
        function_name = functionality.get("function_name", "execute")
        inputs = functionality.get("inputs", {})
        outputs = functionality.get("outputs", {})
        logic = functionality.get("logic", "")
        
        # Build input/output schemas
        input_schema = self._build_schema_from_spec(inputs)
        output_schema = self._build_schema_from_spec(outputs)
        
        # Generate Python code
        code = self._generate_python_function(
            function_name, inputs, outputs, logic
        )
        
        # Validate the code
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.logger.error(f"Generated invalid Python code: {e}")
            # Try to fix common issues
            code = self._fix_python_syntax(code)
        
        return ToolBuilder.create_python_tool(
            name=name,
            description=description,
            code=code,
            function_name=function_name,
            input_schema=input_schema,
            output_schema=output_schema
        )
    
    async def _create_http_tool(
        self, 
        name: str, 
        description: str, 
        functionality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an HTTP-based tool."""
        url = functionality.get("url", "")
        method = functionality.get("method", "POST")
        headers = functionality.get("headers", {})
        inputs = functionality.get("inputs", {})
        outputs = functionality.get("outputs", {})
        
        # Build schemas
        input_schema = self._build_schema_from_spec(inputs)
        output_schema = self._build_schema_from_spec(outputs)
        
        return ToolBuilder.create_http_tool(
            name=name,
            description=description,
            url=url,
            method=method,
            headers=headers,
            input_schema=input_schema,
            output_schema=output_schema
        )
    
    async def _generate_code(self, task: Task) -> TaskResult:
        """Generate code based on requirements."""
        language = task.input_data.get("language", "python")
        requirements = task.input_data.get("requirements", "")
        context = task.input_data.get("context", {})
        
        self.logger.info(f"Generating {language} code for: {requirements}")
        
        try:
            if language == "python":
                code = self._generate_python_code(requirements, context)
            elif language == "javascript":
                code = self._generate_javascript_code(requirements, context)
            else:
                code = f"# Code generation for {language} not implemented\n# Requirements: {requirements}"
            
            return TaskResult(
                success=True,
                output={
                    "code": code,
                    "language": language,
                    "requirements": requirements
                },
                metadata={"lines_of_code": len(code.split('\n'))}
            )
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _handle_general_coding_task(self, task: Task) -> TaskResult:
        """Handle general coding tasks."""
        # Analyze the task to determine the best approach
        task_analysis = self._analyze_coding_task(task)
        
        if task_analysis["requires_tool"]:
            # Create a tool for this task
            tool_spec = {
                "name": f"{task.name.replace(' ', '_').lower()}",
                "description": task.description,
                "type": task_analysis["tool_type"],
                "functionality": task_analysis["functionality"]
            }
            
            create_task = Task(
                type=TaskType.TOOL_CREATION,
                name=f"Create tool for {task.name}",
                description=f"Auto-create tool for: {task.description}",
                created_by=self.id,
                input_data=tool_spec
            )
            
            return await self._create_tool(create_task)
        else:
            # Generate code directly
            code_task = Task(
                type=TaskType.CODE_GENERATION,
                name=task.name,
                description=task.description,
                created_by=self.id,
                input_data={
                    "language": task_analysis["language"],
                    "requirements": task.description,
                    "context": task.input_data
                }
            )
            
            return await self._generate_code(code_task)
    
    def _analyze_coding_task(self, task: Task) -> Dict[str, Any]:
        """Analyze a task to determine coding approach."""
        description_lower = task.description.lower()
        
        # Simple heuristics for task analysis
        requires_tool = any(keyword in description_lower for keyword in [
            "api", "service", "endpoint", "integration", "automate", "monitor"
        ])
        
        tool_type = "http" if any(keyword in description_lower for keyword in [
            "api", "rest", "http", "request", "webhook"
        ]) else "python"
        
        language = "javascript" if "javascript" in description_lower or "js" in description_lower else "python"
        
        return {
            "requires_tool": requires_tool,
            "tool_type": tool_type,
            "language": language,
            "functionality": self._extract_functionality(task)
        }
    
    def _extract_functionality(self, task: Task) -> Dict[str, Any]:
        """Extract functionality requirements from task."""
        return {
            "inputs": task.input_data.get("inputs", {}),
            "outputs": task.input_data.get("outputs", {}),
            "logic": task.description,
            "url": task.input_data.get("url", ""),
            "method": task.input_data.get("method", "POST")
        }
    
    def _generate_python_function(
        self, 
        function_name: str, 
        inputs: Dict[str, Any], 
        outputs: Dict[str, Any], 
        logic: str
    ) -> str:
        """Generate a Python function based on specifications."""
        # Build parameter list
        params = []
        for param_name, param_info in inputs.items():
            param_type = param_info.get("type", "Any")
            params.append(f"{param_name}: {self._python_type(param_type)}")
        
        param_list = ", ".join(params) if params else ""
        
        # Generate function code
        code = f'''
def {function_name}({param_list}) -> dict:
    """
    {logic}
    """
    try:
        # Implementation based on requirements
        result = {{}}
        
        # Process inputs
        {self._generate_input_processing(inputs)}
        
        # Core logic
        {self._generate_core_logic(logic, inputs, outputs)}
        
        # Format output
        {self._generate_output_formatting(outputs)}
        
        return result
    except Exception as e:
        return {{"error": str(e)}}
'''
        return textwrap.dedent(code).strip()
    
    def _generate_python_code(self, requirements: str, context: Dict[str, Any]) -> str:
        """Generate Python code for general requirements."""
        template = self.code_templates.get("python_general", "")
        
        # Simple template-based generation
        code = f'''
# Auto-generated Python code
# Requirements: {requirements}

import json
from typing import Dict, Any, List, Optional

def main(**kwargs) -> Dict[str, Any]:
    """
    Main function implementing the requirements.
    """
    result = {{}}
    
    # TODO: Implement based on requirements
    # Context: {json.dumps(context, indent=2)}
    
    return result

if __name__ == "__main__":
    # Example usage
    result = main()
    print(json.dumps(result, indent=2))
'''
        return code.strip()
    
    def _generate_javascript_code(self, requirements: str, context: Dict[str, Any]) -> str:
        """Generate JavaScript code for requirements."""
        code = f'''
// Auto-generated JavaScript code
// Requirements: {requirements}

function main(params) {{
    // TODO: Implement based on requirements
    // Context: {json.dumps(context, indent=2)}
    
    return {{
        success: true,
        data: {{}}
    }};
}}

// Export for use as a tool
module.exports = {{ main }};
'''
        return code.strip()
    
    def _build_schema_from_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON schema from specification."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for field_name, field_info in spec.items():
            field_type = field_info.get("type", "string")
            field_desc = field_info.get("description", "")
            is_required = field_info.get("required", False)
            
            schema["properties"][field_name] = {
                "type": self._json_type(field_type),
                "description": field_desc
            }
            
            if is_required:
                schema["required"].append(field_name)
        
        return schema
    
    def _python_type(self, type_str: str) -> str:
        """Convert type string to Python type."""
        type_map = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]"
        }
        return type_map.get(type_str, "Any")
    
    def _json_type(self, type_str: str) -> str:
        """Convert type string to JSON schema type."""
        type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object"
        }
        return type_map.get(type_str.lower(), type_str)
    
    def _generate_input_processing(self, inputs: Dict[str, Any]) -> str:
        """Generate code for processing inputs."""
        if not inputs:
            return "# No input processing needed"
        
        lines = []
        for param_name, param_info in inputs.items():
            lines.append(f"# Process {param_name}")
            lines.append(f"processed_{param_name} = {param_name}")
        
        return "\n        ".join(lines)
    
    def _generate_core_logic(self, logic: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
        """Generate core logic implementation."""
        return f'''# Core logic implementation
        # TODO: Implement logic based on: {logic}
        
        # Example implementation
        for key, value in kwargs.items():
            result[key] = value'''
    
    def _generate_output_formatting(self, outputs: Dict[str, Any]) -> str:
        """Generate code for formatting outputs."""
        if not outputs:
            return "# Return raw result"
        
        lines = ["# Format output according to schema"]
        for output_name, output_info in outputs.items():
            lines.append(f"result['{output_name}'] = result.get('{output_name}', None)")
        
        return "\n        ".join(lines)
    
    def _fix_python_syntax(self, code: str) -> str:
        """Attempt to fix common Python syntax errors."""
        # Simple fixes
        fixes = [
            ("    \n", "\n"),  # Remove empty indented lines
            ("\t", "    "),    # Replace tabs with spaces
            (":\n\n", ":\n    pass\n\n"),  # Add pass to empty blocks
        ]
        
        for old, new in fixes:
            code = code.replace(old, new)
        
        return code
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code generation templates."""
        return {
            "python_general": "# Python template",
            "javascript_general": "// JavaScript template",
            "api_client": "# API client template",
            "data_processor": "# Data processor template"
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.tool_registry.close()