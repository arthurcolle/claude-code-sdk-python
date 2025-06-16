# Advanced Conversation Features

This guide covers advanced conversation features that build upon the basic multi-turn functionality, providing powerful tools for managing complex interactions with Claude.

## Table of Contents

1. [Conversation Manager](#conversation-manager)
2. [Conversation Templates](#conversation-templates)
3. [Conversation Chains](#conversation-chains)
4. [Advanced Workflows](#advanced-workflows)
5. [Best Practices](#best-practices)

## Conversation Manager

The `ConversationManager` class provides persistent conversation management with advanced features like branching, tagging, and metadata.

### Basic Usage

```python
from claude_code_sdk.conversation_manager import ConversationManager

# Initialize manager with optional storage path
manager = ConversationManager()

# Create a new conversation
session_id, messages = await manager.create_conversation(
    initial_prompt="Let's build a web application",
    metadata={"project": "my-app", "version": "1.0"},
    tags=["web", "development"]
)

async for message in messages:
    # Process messages
    pass
```

### Key Features

#### 1. Conversation Persistence

Conversations are automatically saved to disk:

```python
# Conversations persist between sessions
manager = ConversationManager(Path.home() / ".my_conversations")

# List all conversations
conversations = manager.list_conversations()
for conv in conversations:
    print(f"{conv.session_id}: {conv.tags}")
```

#### 2. Conversation Branching

Create alternative paths from any point:

```python
# Create a branch to explore different approaches
branch_id, branch_messages = await manager.branch_conversation(
    parent_session_id=session_id,
    prompt="Let's try a different architecture"
)

# The branch maintains parent context but creates new session
async for message in branch_messages:
    pass
```

#### 3. Metadata and Tagging

Organize conversations with metadata and tags:

```python
# Add tags for organization
manager.tag_conversation(session_id, ["priority-high", "backend"])

# Add metadata
manager.add_metadata(session_id, {
    "reviewed_by": "team-lead",
    "status": "in-progress"
})

# Filter conversations
recent_backend = manager.list_conversations(
    tags=["backend"],
    since=datetime.now() - timedelta(days=7)
)
```

#### 4. Conversation Export

Export conversations for sharing or backup:

```python
# Export single conversation
manager.export_conversation(session_id, Path("conversation.json"))

# Export includes full history and metadata
```

#### 5. Conversation Trees

Visualize branching conversations:

```python
# Get conversation tree structure
tree = manager.get_conversation_tree(root_session_id)

# Tree shows all branches and their relationships
```

## Conversation Templates

Templates provide pre-configured conversation setups for common tasks.

### Using Built-in Templates

```python
from claude_code_sdk.conversation_templates import TemplateManager

template_manager = TemplateManager()

# Get a template
review_template = template_manager.get_template("code_review")

# Use template to create options and prompt
context = {"file_path": "src/main.py"}
prompt = review_template.format_initial_prompt(context)
options = review_template.create_options()

async for message in query(prompt=prompt, options=options):
    pass
```

### Available Templates

1. **Code Review** - Comprehensive code analysis
2. **Debugging** - Systematic bug investigation
3. **Refactoring** - Safe code restructuring
4. **Testing** - Test generation and improvement
5. **Documentation** - Doc creation and updates
6. **Pair Programming** - Collaborative coding
7. **Architecture** - System design discussions
8. **Learning** - Educational conversations

### Creating Custom Templates

```python
custom_template = template_manager.create_custom_template(
    name="Security Audit",
    description="Security vulnerability assessment",
    system_prompt="You are a security expert...",
    initial_prompts=[
        "Audit {file_path} for security vulnerabilities"
    ],
    required_context=["file_path"],
    follow_up_suggestions=[
        "What are the OWASP top 10 risks here?",
        "How can we fix these vulnerabilities?",
    ]
)
```

### Template Discovery

```python
# Suggest template based on task description
task = "I need help debugging a memory leak"
suggested = template_manager.suggest_template(task)
# Returns: debugging template
```

## Conversation Chains

Chains automate multi-step workflows with context passing between steps.

### Creating Chains

```python
from claude_code_sdk.conversation_chains import ConversationChain, ChainStep

chain = ConversationChain(
    name="feature_development",
    steps=[
        ChainStep(
            name="design",
            prompt_template="Design a {feature_type} feature",
            result_processor=lambda msgs: {"design": extract_text(msgs)}
        ),
        ChainStep(
            name="implement",
            prompt_template="Implement based on design: {design}",
            dependencies=["design"],
            options_modifier=lambda opts: ClaudeCodeOptions(
                allowed_tools=["Write", "Edit"],
                permission_mode="acceptEdits"
            )
        ),
        ChainStep(
            name="test",
            prompt_template="Write tests for the implementation",
            dependencies=["implement"]
        )
    ]
)
```

### Chain Features

#### 1. Conditional Steps

```python
ChainStep(
    name="optimize",
    prompt_template="Optimize the code",
    condition=lambda ctx: ctx.get("performance_score", 100) < 80,
    dependencies=["benchmark"]
)
```

#### 2. Parallel Execution

```python
# Execute independent steps in parallel
result = await chain.execute(
    context_overrides={"feature_type": "authentication"},
    parallel=True
)
```

#### 3. Error Handling

```python
ChainStep(
    name="critical_step",
    prompt_template="Perform critical operation",
    retry_on_failure=True,
    max_retries=3
)
```

#### 4. Context Processing

```python
def process_test_results(messages):
    # Extract test results from messages
    passed = count_passed_tests(messages)
    failed = count_failed_tests(messages)
    return {
        "tests_passed": passed,
        "tests_failed": failed,
        "all_tests_pass": failed == 0
    }

ChainStep(
    name="run_tests",
    prompt_template="Run the test suite",
    result_processor=process_test_results
)
```

### Pre-built Chains

```python
from claude_code_sdk.conversation_chains import (
    create_debugging_chain,
    create_refactoring_chain,
    create_full_development_chain
)

# Use pre-built debugging chain
debug_chain = create_debugging_chain()
result = await debug_chain.execute(
    context_overrides={
        "issue_description": "Function returns None unexpectedly"
    }
)
```

## Advanced Workflows

Combine all features for sophisticated workflows:

### Example: Feature Development with Review

```python
async def develop_feature_with_review(feature_spec: str):
    manager = ConversationManager()
    template_manager = TemplateManager()
    
    # 1. Start with architecture discussion
    arch_template = template_manager.get_template("architecture")
    session_id, _ = await manager.create_conversation(
        initial_prompt=arch_template.format_initial_prompt({
            "system_description": feature_spec,
            "requirements": "scalable, maintainable"
        }),
        tags=["architecture", "feature-development"]
    )
    
    # 2. Branch for implementation
    impl_branch, _ = await manager.branch_conversation(
        session_id,
        "Let's implement the agreed architecture"
    )
    
    # 3. Run development chain in branch
    dev_chain = create_full_development_chain()
    await dev_chain.execute(
        context_overrides={"requirements": feature_spec}
    )
    
    # 4. Create review branch
    review_branch, _ = await manager.branch_conversation(
        impl_branch,
        "Please review the implementation"
    )
    
    # 5. Use code review template
    review_template = template_manager.get_template("code_review")
    async for msg in manager.continue_conversation(
        review_branch,
        review_template.format_initial_prompt({"file_path": "src/"})
    ):
        pass
    
    # 6. Export all branches
    for branch in [session_id, impl_branch, review_branch]:
        manager.export_conversation(
            branch,
            Path(f"feature_dev_{branch}.json")
        )
```

### Example: Iterative Debugging

```python
async def iterative_debugging(bug_report: str):
    manager = ConversationManager()
    
    # Start debugging session
    session_id, _ = await manager.create_conversation(
        initial_prompt=f"Bug report: {bug_report}",
        tags=["debugging", "bug-fix"]
    )
    
    attempts = []
    max_attempts = 3
    
    for i in range(max_attempts):
        # Branch for each attempt
        attempt_id, _ = await manager.branch_conversation(
            session_id,
            f"Attempt {i+1}: Let's try a different approach"
        )
        
        # Run debugging chain
        debug_chain = create_debugging_chain()
        result = await debug_chain.execute()
        
        if result.status == ChainStatus.COMPLETED:
            # Verify fix in new branch
            verify_id, _ = await manager.branch_conversation(
                attempt_id,
                "Verify the fix works correctly"
            )
            
            success = await verify_fix(manager, verify_id)
            if success:
                return attempt_id
        
        attempts.append(attempt_id)
    
    # Compare all attempts
    return compare_debugging_attempts(manager, attempts)
```

## Best Practices

### 1. Session Management

- Store session IDs for long-running projects
- Use meaningful tags and metadata
- Export important conversations regularly
- Clean up old sessions periodically

### 2. Template Usage

- Start with templates for common tasks
- Customize templates for your workflow
- Share templates with your team
- Version control custom templates

### 3. Chain Design

- Keep steps focused and atomic
- Use dependencies to ensure correct order
- Add error handling for critical steps
- Process results to build context

### 4. Performance Optimization

- Use parallel execution when possible
- Branch instead of starting new conversations
- Reuse sessions for related tasks
- Monitor costs with conversation manager

### 5. Error Recovery

```python
try:
    result = await chain.execute()
except Exception as e:
    # Recover by branching from last good state
    recovery_branch, _ = await manager.branch_conversation(
        last_good_session,
        "Let's recover from the error and try again"
    )
```

### 6. Conversation Hygiene

```python
# Periodically clean up
old_conversations = manager.list_conversations(
    since=datetime.now() - timedelta(days=30)
)

for conv in old_conversations:
    if "archive" not in conv.tags:
        manager.tag_conversation(conv.session_id, ["archive"])
```

## Integration Examples

### With Version Control

```python
async def code_review_with_git(pr_number: int):
    # Get PR details
    pr_info = await get_pr_info(pr_number)
    
    # Create review session
    session_id, _ = await manager.create_conversation(
        initial_prompt=f"Review PR #{pr_number}: {pr_info.title}",
        metadata={
            "pr_number": pr_number,
            "author": pr_info.author,
            "branch": pr_info.branch
        },
        tags=["code-review", "pr"]
    )
    
    # Review each file
    for file in pr_info.changed_files:
        await manager.continue_conversation(
            session_id,
            f"Review changes in {file}"
        )
```

### With CI/CD

```python
async def automated_testing_workflow(commit_hash: str):
    chain = ConversationChain(
        name="ci_testing",
        steps=[
            ChainStep(
                name="analyze_changes",
                prompt_template="Analyze changes in commit {commit_hash}"
            ),
            ChainStep(
                name="generate_tests",
                prompt_template="Generate tests for the changes",
                dependencies=["analyze_changes"]
            ),
            ChainStep(
                name="run_tests",
                prompt_template="Run the generated tests",
                dependencies=["generate_tests"]
            )
        ]
    )
    
    result = await chain.execute(
        context_overrides={"commit_hash": commit_hash}
    )
    
    # Report results to CI system
    return result
```

## Conclusion

These advanced features enable sophisticated conversation management:

- **ConversationManager**: Persistent, branching conversations with metadata
- **Templates**: Pre-configured setups for common tasks
- **Chains**: Automated multi-step workflows

Combine these tools to build powerful applications that maintain context, automate workflows, and provide consistent interactions with Claude.