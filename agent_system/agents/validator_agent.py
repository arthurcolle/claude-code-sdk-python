"""
Validator Agent - Specializes in validating tools and code quality.
"""

import asyncio
import ast
import json
from typing import Dict, Any, List, Optional
import re

from ..core.base_agent import BaseAgent
from ..core.task import Task, TaskResult, TaskType
from ..config import AgentRole, AgentCapability
from ..integrations import get_tool_registry_client


class ValidatorAgent(BaseAgent):
    """Agent specialized in validation and quality assurance."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.VALIDATOR
        self.capabilities = {
            AgentCapability.TOOL_VALIDATION,
            AgentCapability.DATA_ANALYSIS
        }
        self.tool_registry = get_tool_registry_client()
        
        # Validation rules and patterns
        self.validation_rules = self._load_validation_rules()
        self.security_patterns = self._load_security_patterns()
        self.validated_tools: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the validator agent."""
        self.logger.info("Initializing Validator Agent")
        
        # Start periodic validation checks
        asyncio.create_task(self._periodic_validation_check())
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process validation tasks."""
        self.logger.info(f"Processing validation task: {task.name}")
        
        try:
            if task.type == TaskType.TOOL_VALIDATION:
                return await self._validate_tool(task)
            else:
                return await self._perform_general_validation(task)
                
        except Exception as e:
            self.logger.error(f"Error processing task {task.id}: {e}")
            return TaskResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _validate_tool(self, task: Task) -> TaskResult:
        """Validate a tool for security, functionality, and quality."""
        tool_id = task.input_data.get("tool_id")
        tool_data = task.input_data.get("tool_data")
        validation_level = task.input_data.get("level", "standard")
        
        self.logger.info(f"Validating tool {tool_id} at {validation_level} level")
        
        # Get tool data if not provided
        if not tool_data and tool_id:
            try:
                tool_data = await self.tool_registry.get_tool_by_id(tool_id)
            except Exception as e:
                return TaskResult(
                    success=False,
                    output=None,
                    error=f"Failed to fetch tool: {e}"
                )
        
        if not tool_data:
            return TaskResult(
                success=False,
                output=None,
                error="No tool data provided"
            )
        
        # Perform validation checks
        validation_results = {
            "tool_id": tool_id or tool_data.get("id"),
            "tool_name": tool_data.get("name"),
            "validation_level": validation_level,
            "checks": {},
            "issues": [],
            "recommendations": [],
            "overall_score": 0.0
        }
        
        # 1. Schema validation
        schema_result = self._validate_schemas(tool_data)
        validation_results["checks"]["schema"] = schema_result
        
        # 2. Security validation
        security_result = await self._validate_security(tool_data)
        validation_results["checks"]["security"] = security_result
        
        # 3. Code validation (if applicable)
        if tool_data.get("action", {}).get("type") == "python":
            code_result = self._validate_python_code(tool_data)
            validation_results["checks"]["code"] = code_result
        
        # 4. Functionality validation
        func_result = await self._validate_functionality(tool_data)
        validation_results["checks"]["functionality"] = func_result
        
        # Calculate overall score
        scores = []
        for check_name, check_result in validation_results["checks"].items():
            if isinstance(check_result, dict) and "score" in check_result:
                scores.append(check_result["score"])
                validation_results["issues"].extend(check_result.get("issues", []))
                validation_results["recommendations"].extend(
                    check_result.get("recommendations", [])
                )
        
        validation_results["overall_score"] = (
            sum(scores) / len(scores) if scores else 0.0
        )
        
        # Determine validation status
        is_approved = (
            validation_results["overall_score"] >= 0.8 and
            not any(issue["severity"] == "critical" for issue in validation_results["issues"])
        )
        
        # Update tool status if auto-validation is enabled
        if tool_id and task.input_data.get("auto_update", False):
            try:
                await self.tool_registry.validate_tool(
                    tool_id,
                    "approved" if is_approved else "rejected"
                )
                validation_results["status_updated"] = True
            except Exception as e:
                self.logger.error(f"Failed to update tool status: {e}")
                validation_results["status_updated"] = False
        
        # Cache validation results
        self.validated_tools[tool_id or tool_data.get("name")] = validation_results
        
        return TaskResult(
            success=True,
            output=validation_results,
            metadata={
                "approved": is_approved,
                "score": validation_results["overall_score"]
            }
        )
    
    def _validate_schemas(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input and output schemas."""
        result = {
            "score": 1.0,
            "issues": [],
            "recommendations": []
        }
        
        input_schema = tool_data.get("input_schema", {})
        output_schema = tool_data.get("output_schema", {})
        
        # Check if schemas are present
        if not input_schema:
            result["issues"].append({
                "type": "missing_schema",
                "severity": "high",
                "message": "Input schema is missing"
            })
            result["score"] -= 0.3
        
        if not output_schema:
            result["issues"].append({
                "type": "missing_schema",
                "severity": "medium",
                "message": "Output schema is missing"
            })
            result["score"] -= 0.2
        
        # Validate schema structure
        for schema_name, schema in [("input", input_schema), ("output", output_schema)]:
            if schema and isinstance(schema, dict):
                # Check for type
                if "type" not in schema:
                    result["issues"].append({
                        "type": "invalid_schema",
                        "severity": "medium",
                        "message": f"{schema_name} schema missing 'type' field"
                    })
                    result["score"] -= 0.1
                
                # Check for properties if type is object
                if schema.get("type") == "object" and "properties" not in schema:
                    result["recommendations"].append(
                        f"Add 'properties' field to {schema_name} schema"
                    )
        
        return result
    
    async def _validate_security(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool for security issues."""
        result = {
            "score": 1.0,
            "issues": [],
            "recommendations": []
        }
        
        action = tool_data.get("action", {})
        action_type = action.get("type")
        
        if action_type == "python":
            # Check Python code for security issues
            code = action.get("python", {}).get("code", "")
            
            # Check for dangerous patterns
            for pattern, description in self.security_patterns["python"].items():
                if re.search(pattern, code, re.IGNORECASE):
                    result["issues"].append({
                        "type": "security_risk",
                        "severity": "critical",
                        "message": f"Potential security risk: {description}"
                    })
                    result["score"] -= 0.5
            
            # Check for imports
            dangerous_imports = ["os", "subprocess", "eval", "exec", "__import__"]
            for imp in dangerous_imports:
                if imp in code:
                    result["issues"].append({
                        "type": "dangerous_import",
                        "severity": "high",
                        "message": f"Use of potentially dangerous module: {imp}"
                    })
                    result["score"] -= 0.3
        
        elif action_type == "http":
            # Check HTTP configuration
            http_config = action.get("http", {})
            url = http_config.get("url", "")
            
            # Check for HTTPS
            if url and not url.startswith("https://"):
                result["issues"].append({
                    "type": "insecure_protocol",
                    "severity": "medium",
                    "message": "HTTP endpoint should use HTTPS"
                })
                result["score"] -= 0.2
            
            # Check for sensitive data in headers
            headers = http_config.get("headers", {})
            for key, value in headers.items():
                if any(sensitive in key.lower() for sensitive in ["api", "key", "token", "secret"]):
                    if "{" not in value:  # Not a template
                        result["issues"].append({
                            "type": "exposed_credential",
                            "severity": "critical",
                            "message": f"Potential exposed credential in header: {key}"
                        })
                        result["score"] -= 0.5
        
        return result
    
    def _validate_python_code(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Python code syntax and quality."""
        result = {
            "score": 1.0,
            "issues": [],
            "recommendations": []
        }
        
        code = tool_data.get("action", {}).get("python", {}).get("code", "")
        
        if not code:
            result["issues"].append({
                "type": "missing_code",
                "severity": "critical",
                "message": "Python code is empty"
            })
            result["score"] = 0.0
            return result
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            result["issues"].append({
                "type": "syntax_error",
                "severity": "critical",
                "message": f"Python syntax error: {e}"
            })
            result["score"] -= 0.5
            return result
        
        # Code quality checks
        lines = code.split('\n')
        
        # Check for docstrings
        if 'def ' in code and '"""' not in code and "'''" not in code:
            result["recommendations"].append("Add docstrings to functions")
            result["score"] -= 0.1
        
        # Check for error handling
        if 'try:' not in code and 'except' not in code:
            result["recommendations"].append("Consider adding error handling")
            result["score"] -= 0.1
        
        # Check line length
        long_lines = [i for i, line in enumerate(lines) if len(line) > 100]
        if long_lines:
            result["recommendations"].append(
                f"Lines {long_lines[:3]} exceed 100 characters"
            )
        
        # Check for type hints
        if 'def ' in code and '->' not in code:
            result["recommendations"].append("Consider adding type hints")
        
        return result
    
    async def _validate_functionality(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool functionality through test execution."""
        result = {
            "score": 1.0,
            "issues": [],
            "recommendations": []
        }
        
        # Generate test inputs based on schema
        test_inputs = self._generate_test_inputs(tool_data.get("input_schema", {}))
        
        if not test_inputs:
            result["recommendations"].append(
                "Unable to generate test inputs from schema"
            )
            return result
        
        # Try to execute the tool with test inputs
        try:
            test_result = await self.tool_registry.execute_tool(
                tool_name=tool_data.get("name"),
                input_data=test_inputs[0]  # Use first test case
            )
            
            # Check if output matches schema
            output_schema = tool_data.get("output_schema", {})
            if output_schema:
                validation_errors = self._validate_against_schema(
                    test_result, output_schema
                )
                if validation_errors:
                    result["issues"].append({
                        "type": "output_mismatch",
                        "severity": "high",
                        "message": f"Output doesn't match schema: {validation_errors}"
                    })
                    result["score"] -= 0.3
            
        except Exception as e:
            result["issues"].append({
                "type": "execution_error",
                "severity": "high",
                "message": f"Tool execution failed: {str(e)}"
            })
            result["score"] -= 0.5
        
        return result
    
    async def _perform_general_validation(self, task: Task) -> TaskResult:
        """Perform general validation based on task requirements."""
        validation_type = task.input_data.get("type", "data")
        target = task.input_data.get("target")
        rules = task.input_data.get("rules", {})
        
        self.logger.info(f"Performing {validation_type} validation")
        
        validation_result = {
            "type": validation_type,
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if validation_type == "data":
            # Validate data structure
            for rule_name, rule_config in rules.items():
                if not self._check_rule(target, rule_config):
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Rule '{rule_name}' failed"
                    )
        
        return TaskResult(
            success=True,
            output=validation_result,
            metadata={"validation_type": validation_type}
        )
    
    def _generate_test_inputs(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test inputs based on schema."""
        if not schema or schema.get("type") != "object":
            return []
        
        test_cases = []
        base_case = {}
        
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Generate base case with required fields
        for prop_name, prop_schema in properties.items():
            if prop_name in required:
                base_case[prop_name] = self._generate_value_for_type(prop_schema)
        
        test_cases.append(base_case)
        
        # Generate additional test cases with optional fields
        full_case = base_case.copy()
        for prop_name, prop_schema in properties.items():
            if prop_name not in required:
                full_case[prop_name] = self._generate_value_for_type(prop_schema)
        
        if full_case != base_case:
            test_cases.append(full_case)
        
        return test_cases
    
    def _generate_value_for_type(self, schema: Dict[str, Any]) -> Any:
        """Generate a test value based on schema type."""
        schema_type = schema.get("type", "string")
        
        if schema_type == "string":
            return schema.get("default", "test_value")
        elif schema_type == "integer":
            return schema.get("default", 42)
        elif schema_type == "number":
            return schema.get("default", 3.14)
        elif schema_type == "boolean":
            return schema.get("default", True)
        elif schema_type == "array":
            return []
        elif schema_type == "object":
            return {}
        else:
            return None
    
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate data against a JSON schema."""
        errors = []
        
        # Simple validation - would use jsonschema library in production
        schema_type = schema.get("type")
        
        if schema_type == "object" and not isinstance(data, dict):
            errors.append(f"Expected object, got {type(data).__name__}")
        elif schema_type == "array" and not isinstance(data, list):
            errors.append(f"Expected array, got {type(data).__name__}")
        elif schema_type == "string" and not isinstance(data, str):
            errors.append(f"Expected string, got {type(data).__name__}")
        elif schema_type == "number" and not isinstance(data, (int, float)):
            errors.append(f"Expected number, got {type(data).__name__}")
        elif schema_type == "boolean" and not isinstance(data, bool):
            errors.append(f"Expected boolean, got {type(data).__name__}")
        
        # Check required fields for objects
        if schema_type == "object" and isinstance(data, dict):
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
        
        return errors
    
    def _check_rule(self, target: Any, rule_config: Dict[str, Any]) -> bool:
        """Check if target satisfies a validation rule."""
        rule_type = rule_config.get("type")
        
        if rule_type == "required":
            return target is not None
        elif rule_type == "type":
            expected_type = rule_config.get("value")
            return type(target).__name__ == expected_type
        elif rule_type == "min_length":
            min_len = rule_config.get("value", 0)
            return len(str(target)) >= min_len
        elif rule_type == "pattern":
            pattern = rule_config.get("value", "")
            return bool(re.match(pattern, str(target)))
        else:
            return True
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules."""
        return {
            "schema": {
                "required_fields": ["type", "properties"],
                "valid_types": ["object", "array", "string", "number", "boolean", "null"]
            },
            "code": {
                "max_line_length": 100,
                "required_docstring": True,
                "required_type_hints": False
            }
        }
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load security validation patterns."""
        return {
            "python": {
                r"eval\s*\(": "Use of eval() function",
                r"exec\s*\(": "Use of exec() function",
                r"__import__": "Dynamic import",
                r"pickle\.loads": "Unsafe deserialization",
                r"subprocess\.call.*shell\s*=\s*True": "Shell injection risk",
                r"os\.system": "Direct system command execution"
            },
            "general": {
                r"password|secret|api_key": "Potential hardcoded credential",
                r"TODO|FIXME|XXX": "Incomplete implementation"
            }
        }
    
    async def _periodic_validation_check(self):
        """Periodically check and validate pending tools."""
        while self._running:
            try:
                # Get pending tools
                pending_tools = await self.tool_registry.get_pending_tools()
                
                if pending_tools:
                    self.logger.info(f"Found {len(pending_tools)} pending tools to validate")
                    
                    for tool in pending_tools[:5]:  # Process up to 5 at a time
                        # Create validation task
                        validation_task = Task(
                            type=TaskType.TOOL_VALIDATION,
                            name=f"Validate {tool['name']}",
                            description=f"Validate pending tool: {tool['name']}",
                            created_by=self.id,
                            input_data={
                                "tool_id": tool["id"],
                                "tool_data": tool,
                                "level": "standard",
                                "auto_update": True
                            }
                        )
                        
                        # Process validation
                        await self._execute_task(validation_task)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in periodic validation: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.tool_registry.close()