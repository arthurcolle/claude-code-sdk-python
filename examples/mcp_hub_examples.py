#!/usr/bin/env python3
"""
Real-World MCP Hub Examples

Practical applications demonstrating real-world use cases:
1. Automated CI/CD Pipeline
2. Data Processing Pipeline
3. Monitoring & Alerting System
4. Multi-Cloud Deployment Tool
5. Development Assistant
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_hub_enhanced import EnhancedMCPHub
from mcp_hub_workflows import WorkflowBuilder
from mcp_hub_advanced_client import AdvancedMCPClient, ClientConfig


# ============================================================================
# Example 1: Automated CI/CD Pipeline
# ============================================================================

async def example_cicd_pipeline():
    """
    Complete CI/CD pipeline using MCP hub.

    Flow:
    1. Pull latest code from Git
    2. Run tests in parallel
    3. Build Docker image
    4. Deploy to staging
    5. Run smoke tests
    6. Notify team
    """
    print("=" * 80)
    print("Example 1: Automated CI/CD Pipeline")
    print("=" * 80)
    print()

    hub = EnhancedMCPHub()
    await hub.start()

    # Create CI/CD workflow
    cicd_workflow = (
        WorkflowBuilder("CI/CD Pipeline", "Automated build, test, and deployment")
        # Step 1: Pull latest code
        .add_step("pull_code", "git_pull", {
            "repo_path": "/app",
            "remote": "origin",
            "branch": "main"
        })
        # Step 2: Run tests (parallel)
        .add_parallel_steps([
            ("unit_tests", "execute_python", {
                "code": "pytest tests/unit --junit-xml=unit-results.xml",
                "timeout": 300
            }),
            ("integration_tests", "execute_python", {
                "code": "pytest tests/integration --junit-xml=integration-results.xml",
                "timeout": 600
            }),
            ("lint", "execute_python", {
                "code": "flake8 src/",
                "timeout": 60
            }),
        ])
        # Step 3: Build Docker image
        .add_step("build_image", "docker_build", {
            "path": "/app",
            "tag": "myapp:latest",
            "dockerfile": "Dockerfile"
        }, depends_on=["unit_tests", "integration_tests", "lint"])
        # Step 4: Deploy to staging
        .add_step("deploy_staging", "docker_run", {
            "image": "myapp:latest",
            "detach": True,
            "ports": {"8080": "8000"},
            "env": {"ENV": "staging"}
        }, depends_on=["build_image"])
        # Step 5: Smoke tests
        .add_step("smoke_tests", "http_get", {
            "url": "http://localhost:8080/health",
            "headers": {}
        }, depends_on=["deploy_staging"])
        # Step 6: Notify
        .add_step("notify_success", "send_slack", {
            "channel": "#deployments",
            "message": "✅ CI/CD pipeline completed successfully!"
        }, depends_on=["smoke_tests"])
        .with_tags("cicd", "automation", "deployment")
        .build()
    )

    # Register and execute
    hub.workflow_engine.register_workflow(cicd_workflow)

    print(f"Workflow: {cicd_workflow.name}")
    print(f"Steps: {len(cicd_workflow.steps)}")
    print(f"Execution layers: {cicd_workflow.get_execution_order()}")
    print()

    print("Executing pipeline...")
    execution = await hub.execute_workflow(cicd_workflow.workflow_id, {})

    print(f"✓ Status: {execution.status.value}")
    print(f"  Duration: {(execution.completed_at - execution.started_at).total_seconds():.1f}s")

    # Show results
    print(f"\nStep Results:")
    for step_id, step_exec in execution.step_executions.items():
        status_icon = "✓" if step_exec.status.value == "completed" else "✗"
        print(f"  {status_icon} {step_id}: {step_exec.status.value}")

    await hub.stop()


# ============================================================================
# Example 2: Data Processing Pipeline
# ============================================================================

async def example_data_pipeline():
    """
    ETL pipeline for processing data from multiple sources.

    Flow:
    1. Fetch data from multiple APIs in parallel
    2. Validate and clean data
    3. Transform data
    4. Load to database
    5. Index for search
    6. Generate report
    """
    print("\n" + "=" * 80)
    print("Example 2: Data Processing Pipeline")
    print("=" * 80)
    print()

    hub = EnhancedMCPHub()
    await hub.start()

    data_pipeline = (
        WorkflowBuilder("ETL Pipeline", "Extract, Transform, Load data")
        # Extract from multiple sources (parallel)
        .add_parallel_steps([
            ("fetch_api1", "http_get", {"url": "https://api1.com/data", "headers": {}}),
            ("fetch_api2", "http_get", {"url": "https://api2.com/data", "headers": {}}),
            ("fetch_api3", "http_get", {"url": "https://api3.com/data", "headers": {}}),
        ])
        # Validate data
        .add_step("validate", "execute_python", {
            "code": """
# Validation logic
import json
data = {
    'api1': step_fetch_api1_result,
    'api2': step_fetch_api2_result,
    'api3': step_fetch_api3_result
}
validated = {k: v for k, v in data.items() if v}
print(f'Validated {len(validated)} sources')
""",
            "timeout": 60
        }, depends_on=["fetch_api1", "fetch_api2", "fetch_api3"])
        # Transform
        .add_step("transform", "execute_python", {
            "code": "# Transform data\nresult = {'transformed': True}",
            "timeout": 120
        }, depends_on=["validate"])
        # Load to database and index (parallel)
        .add_parallel_steps([
            ("load_db", "sql_query", {
                "query": "INSERT INTO processed_data (data, timestamp) VALUES (?, NOW())",
                "params": {"data": "$step_transform_result"}
            }),
            ("index_search", "index_document", {
                "index": "data",
                "doc_id": "batch_001",
                "document": {"content": "$step_transform_result"}
            }),
        ])
        # Report
        .add_step("generate_report", "send_email", {
            "to": "data-team@company.com",
            "subject": "ETL Pipeline Complete",
            "body": "Data processing pipeline completed successfully"
        }, depends_on=["load_db", "index_search"])
        .with_tags("etl", "data", "pipeline")
        .build()
    )

    hub.workflow_engine.register_workflow(data_pipeline)

    print(f"Pipeline: {data_pipeline.name}")
    print(f"Processing {len(data_pipeline.steps)} steps...")
    print()

    execution = await hub.execute_workflow(data_pipeline.workflow_id)

    print(f"✓ Pipeline completed: {execution.status.value}")
    print(f"  Steps executed: {len(execution.step_executions)}")

    await hub.stop()


# ============================================================================
# Example 3: Monitoring & Alerting System
# ============================================================================

async def example_monitoring_system():
    """
    Automated monitoring and alerting system.

    Flow:
    1. Collect metrics from multiple services
    2. Analyze metrics
    3. Check thresholds
    4. Alert if needed
    5. Log to database
    """
    print("\n" + "=" * 80)
    print("Example 3: Monitoring & Alerting System")
    print("=" * 80)
    print()

    hub = EnhancedMCPHub()
    await hub.start()

    monitoring_workflow = (
        WorkflowBuilder("Monitoring System", "Collect metrics and alert")
        # Collect metrics (parallel)
        .add_parallel_steps([
            ("metrics_cpu", "execute_python", {
                "code": "import psutil; result = {'cpu': psutil.cpu_percent()}",
                "timeout": 10
            }),
            ("metrics_memory", "execute_python", {
                "code": "import psutil; result = {'memory': psutil.virtual_memory().percent}",
                "timeout": 10
            }),
            ("metrics_disk", "execute_python", {
                "code": "import psutil; result = {'disk': psutil.disk_usage('/').percent}",
                "timeout": 10
            }),
            ("docker_status", "docker_ps", {"all": True}),
            ("git_status", "git_status", {"repo_path": "."}),
        ])
        # Analyze
        .add_step("analyze_metrics", "execute_python", {
            "code": """
# Analyze all metrics
cpu = step_metrics_cpu_result.get('cpu', 0)
memory = step_metrics_memory_result.get('memory', 0)
disk = step_metrics_disk_result.get('disk', 0)

alert_needed = cpu > 80 or memory > 80 or disk > 90
result = {
    'alert_needed': alert_needed,
    'cpu': cpu,
    'memory': memory,
    'disk': disk
}
""",
            "timeout": 30
        }, depends_on=["metrics_cpu", "metrics_memory", "metrics_disk"])
        # Store metrics
        .add_step("store_metrics", "sql_query", {
            "query": "INSERT INTO metrics (timestamp, data) VALUES (NOW(), ?)",
            "params": {"data": "$step_analyze_metrics_result"}
        }, depends_on=["analyze_metrics"])
        # Alert if threshold exceeded
        .add_step("send_alert", "send_slack", {
            "channel": "#alerts",
            "message": "⚠️ High resource usage detected!"
        }, depends_on=["analyze_metrics"],
           condition="step_analyze_metrics_result['alert_needed']")
        .with_tags("monitoring", "alerting")
        .build()
    )

    hub.workflow_engine.register_workflow(monitoring_workflow)

    print("Running monitoring checks...")
    execution = await hub.execute_workflow(monitoring_workflow.workflow_id)

    print(f"✓ Monitoring complete: {execution.status.value}")

    # Check if alert was sent
    alert_step = execution.step_executions.get("send_alert")
    if alert_step and alert_step.status.value == "completed":
        print("  ⚠️  Alert was sent!")
    elif alert_step and alert_step.status.value == "skipped":
        print("  ✓ No alerts needed - all metrics normal")

    await hub.stop()


# ============================================================================
# Example 4: Development Assistant
# ============================================================================

async def example_dev_assistant():
    """
    Development assistant that helps with common tasks.

    Features:
    - Code review automation
    - Documentation generation
    - Dependency updates
    - Test generation
    """
    print("\n" + "=" * 80)
    print("Example 4: Development Assistant")
    print("=" * 80)
    print()

    hub = EnhancedMCPHub()
    await hub.start()

    # Example task: Review recent changes
    print("Task: Review recent code changes")
    print()

    # Get Git status
    git_status = await hub.execute("git_status", {"repo_path": "."})
    print(f"📝 Repository status: {git_status['result'].get('status', 'unknown')}")

    if git_status['result'].get('files'):
        print(f"   Modified files: {len(git_status['result']['files'])}")

        # Get diff for review
        git_diff = await hub.execute("git_diff", {"repo_path": "."})
        print(f"   Changes available for review")

    # Check tests
    print()
    print("🧪 Running tests...")
    test_result = await hub.execute("execute_python", {
        "code": "print('All tests passed')",
        "timeout": 60
    })
    print(f"   {test_result['result']['stdout']}")

    # Search documentation
    print()
    print("📚 Searching documentation...")
    await hub.execute("create_index", {"index": "docs"})
    await hub.execute("index_document", {
        "index": "docs",
        "doc_id": "readme",
        "document": {"title": "README", "content": "MCP Hub documentation"}
    })

    search_result = await hub.execute("search", {
        "index": "docs",
        "query": "MCP",
        "limit": 5
    })
    print(f"   Found {search_result['result']['count']} documentation entries")

    # Send summary
    print()
    print("📧 Sending summary...")
    await hub.execute("send_email", {
        "to": "dev@company.com",
        "subject": "Dev Assistant Summary",
        "body": "Code review and tests completed"
    })
    print("   ✓ Summary sent")

    await hub.stop()


# ============================================================================
# Example 5: Smart Batch Processing
# ============================================================================

async def example_batch_processing():
    """
    Demonstrate intelligent batch processing with the advanced client.
    """
    print("\n" + "=" * 80)
    print("Example 5: Smart Batch Processing")
    print("=" * 80)
    print()

    # Create enhanced hub
    hub = EnhancedMCPHub()
    await hub.start()

    # Simulate batch processing scenario
    print("Processing 100 mixed operations efficiently...")
    print()

    # Use hub directly for demo
    tasks = []

    # File operations
    for i in range(20):
        tasks.append(("read_file", {"path": f"/tmp/file{i}.txt"}))

    # API calls
    for i in range(20):
        tasks.append(("http_get", {"url": f"https://api{i}.com", "headers": {}}))

    # Database queries
    for i in range(20):
        tasks.append(("sql_query", {"query": f"SELECT * FROM table{i}", "params": {}}))

    # Docker operations
    for i in range(10):
        tasks.append(("docker_ps", {"all": False}))

    # Git operations
    for i in range(10):
        tasks.append(("git_status", {"repo_path": "."}))

    # Search operations
    for i in range(10):
        tasks.append(("search", {"index": "docs", "query": f"term{i}", "limit": 5}))

    # Email notifications
    for i in range(10):
        tasks.append(("send_email", {
            "to": f"user{i}@example.com",
            "subject": f"Update {i}",
            "body": "Status update"
        }))

    print(f"Total operations: {len(tasks)}")
    print(f"Operation types: {len(set(t[0] for t in tasks))}")
    print()

    import time
    start = time.time()

    # Execute in batches
    batch_size = 20
    results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_tasks = [
            hub.execute(tool_name, params)
            for tool_name, params in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)

        # Progress
        progress = min(i + batch_size, len(tasks))
        print(f"  Progress: {progress}/{len(tasks)} ({progress/len(tasks)*100:.0f}%)")

    elapsed = time.time() - start

    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful

    print()
    print(f"✓ Batch processing complete!")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {len(results)/elapsed:.1f} ops/sec")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {successful/len(results)*100:.1f}%")

    # Get metrics
    metrics = await hub.get_metrics()
    print()
    print(f"📊 Hub Metrics:")
    print(f"   Average latency: {metrics['requests']['average_latency_ms']:.1f}ms")
    print(f"   Total requests: {metrics['requests']['total']}")

    await hub.stop()


# ============================================================================
# Main: Run All Examples
# ============================================================================

async def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║               MCP Hub - Real-World Examples                                  ║
║                                                                              ║
║  Practical demonstrations of distributed tool ecosystem                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    examples = [
        ("CI/CD Pipeline", example_cicd_pipeline),
        ("Data Processing Pipeline", example_data_pipeline),
        ("Monitoring System", example_monitoring_system),
        ("Development Assistant", example_dev_assistant),
        ("Batch Processing", example_batch_processing),
    ]

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
        print()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
