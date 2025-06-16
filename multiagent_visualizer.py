#!/usr/bin/env python3
"""
Real-time Visualization and Monitoring for Multi-Agent System
Provides web-based dashboard for monitoring agent activities
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
import aiohttp
from aiohttp import web
import aiohttp_cors

# HTML template for the visualization dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Agent System Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vis-network@latest/dist/vis-network.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/vis-network@latest/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .team-card {
            border-left: 4px solid;
        }
        .team-research { border-left-color: #3498db; }
        .team-development { border-left-color: #2ecc71; }
        .team-analysis { border-left-color: #f39c12; }
        .team-creative { border-left-color: #e74c3c; }
        .team-operations { border-left-color: #9b59b6; }
        .team-quality { border-left-color: #1abc9c; }
        .team-security { border-left-color: #34495e; }
        .team-strategy { border-left-color: #e67e22; }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        #network {
            width: 100%;
            height: 500px;
            border: 1px solid #ddd;
            background-color: white;
            border-radius: 8px;
        }
        .status-active { color: #2ecc71; }
        .status-idle { color: #95a5a6; }
        .status-error { color: #e74c3c; }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .log-container {
            background-color: #2c3e50;
            color: #2ecc71;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            height: 200px;
            overflow-y: auto;
        }
        .log-entry {
            margin: 2px 0;
        }
        .log-time {
            color: #95a5a6;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Multi-Agent System Dashboard</h1>
            <p>Real-time monitoring of 40+ AI agents across 8 specialized teams</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-number" id="total-agents">0</div>
                <div class="stat-label">Total Agents</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="active-agents">0</div>
                <div class="stat-label">Active Agents</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="completed-tasks">0</div>
                <div class="stat-label">Completed Tasks</div>
            </div>
            <div class="stat-box">
                <div class="stat-number" id="system-performance">0%</div>
                <div class="stat-label">System Performance</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Agent Network Visualization</h2>
            <div id="network"></div>
        </div>
        
        <div class="grid" id="team-cards">
            <!-- Team cards will be dynamically inserted here -->
        </div>
        
        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>
        
        <div class="card">
            <h2>System Activity Log</h2>
            <div class="log-container" id="activity-log">
                <!-- Log entries will be added here -->
            </div>
        </div>
    </div>
    
    <script>
        // Initialize network visualization
        const container = document.getElementById('network');
        const nodes = new vis.DataSet();
        const edges = new vis.DataSet();
        const data = { nodes: nodes, edges: edges };
        const options = {
            nodes: {
                shape: 'dot',
                size: 20,
                font: { size: 12 },
                borderWidth: 2
            },
            edges: {
                width: 2,
                arrows: 'to'
            },
            physics: {
                forceAtlas2Based: {
                    gravitationalConstant: -26,
                    centralGravity: 0.005,
                    springLength: 230,
                    springConstant: 0.18
                },
                maxVelocity: 146,
                solver: 'forceAtlas2Based',
                timestep: 0.35,
                stabilization: { iterations: 150 }
            }
        };
        const network = new vis.Network(container, data, options);
        
        // Initialize performance chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:8080/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            // Update stats
            document.getElementById('total-agents').textContent = data.total_agents;
            document.getElementById('active-agents').textContent = data.active_tasks;
            document.getElementById('completed-tasks').textContent = data.completed_tasks;
            document.getElementById('system-performance').textContent = 
                Math.round(data.system_performance * 100) + '%';
            
            // Update team cards
            updateTeamCards(data.teams);
            
            // Update network visualization
            updateNetwork(data.agents, data.messages);
            
            // Update performance chart
            updatePerformanceChart(data.teams);
            
            // Add to activity log
            if (data.latest_activity) {
                addLogEntry(data.latest_activity);
            }
        }
        
        function updateTeamCards(teams) {
            const container = document.getElementById('team-cards');
            container.innerHTML = '';
            
            for (const [teamId, team] of Object.entries(teams)) {
                const card = document.createElement('div');
                card.className = `card team-card team-${team.type}`;
                card.innerHTML = `
                    <h3>${team.name}</h3>
                    <div class="metric">
                        <span>Agents:</span>
                        <span class="metric-value">${team.agents}</span>
                    </div>
                    <div class="metric">
                        <span>Active:</span>
                        <span class="metric-value status-${team.active_agents > 0 ? 'active' : 'idle'}">
                            ${team.active_agents}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Completed Tasks:</span>
                        <span class="metric-value">${team.completed_tasks}</span>
                    </div>
                    <div class="metric">
                        <span>Performance:</span>
                        <span class="metric-value">${Math.round(team.average_performance * 100)}%</span>
                    </div>
                `;
                container.appendChild(card);
            }
        }
        
        function updateNetwork(agents, messages) {
            // Update nodes (agents)
            const nodeUpdates = [];
            for (const [agentId, agent of Object.entries(agents)]) {
                const color = agent.current_task ? '#2ecc71' : '#95a5a6';
                nodeUpdates.push({
                    id: agentId,
                    label: agent.name,
                    title: `${agent.specialization}\\nTeam: ${agent.team}\\nWorkload: ${agent.workload}`,
                    color: color,
                    group: agent.team
                });
            }
            nodes.update(nodeUpdates);
            
            // Update edges (recent communications)
            if (messages && messages.length > 0) {
                const edgeUpdates = messages.slice(-10).map((msg, i) => ({
                    id: `msg_${i}`,
                    from: msg.sender,
                    to: msg.recipient,
                    color: { color: '#3498db', opacity: 0.5 }
                }));
                edges.clear();
                edges.update(edgeUpdates);
            }
        }
        
        function updatePerformanceChart(teams) {
            const teamNames = Object.values(teams).map(t => t.name);
            const performances = Object.values(teams).map(t => t.average_performance);
            
            performanceChart.data.labels = teamNames;
            performanceChart.data.datasets = [{
                label: 'Team Performance',
                data: performances,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                tension: 0.1
            }];
            performanceChart.update();
        }
        
        function addLogEntry(activity) {
            const log = document.getElementById('activity-log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="log-time">[${time}]</span> ${activity}`;
            log.insertBefore(entry, log.firstChild);
            
            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }
        
        // Initial connection message
        addLogEntry('Connected to Multi-Agent System');
    </script>
</body>
</html>
"""


class SystemMonitor:
    """Monitor and visualize the multi-agent system"""
    
    def __init__(self, system):
        self.system = system
        self.app = web.Application()
        self.websockets = set()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/ws', self.websocket_handler)
        self.app.router.add_get('/api/status', self.api_status)
        
        # Configure CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def index(self, request):
        """Serve the dashboard HTML"""
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Send initial status
            status = self.get_enhanced_status()
            await ws.send_json(status)
            
            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'ping':
                        await ws.send_str('pong')
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
        finally:
            self.websockets.remove(ws)
        
        return ws
    
    async def api_status(self, request):
        """API endpoint for system status"""
        status = self.get_enhanced_status()
        return web.json_response(status)
    
    def get_enhanced_status(self):
        """Get enhanced system status with additional monitoring data"""
        base_status = self.system.get_system_status()
        
        # Add agent details
        agents = {}
        for agent_id, agent in self.system.agents.items():
            agents[agent_id] = {
                "name": agent.name,
                "team": agent.team,
                "specialization": agent.specialization,
                "current_task": agent.current_task.title if agent.current_task else None,
                "workload": agent.workload,
                "performance": agent.performance_score,
                "message_count": len(agent.message_queue)
            }
        
        # Add recent messages
        recent_messages = []
        for agent in self.system.agents.values():
            for msg in agent.message_queue[-5:]:
                recent_messages.append({
                    "sender": msg.sender,
                    "recipient": msg.recipient,
                    "type": msg.message_type,
                    "timestamp": msg.timestamp.isoformat()
                })
        
        base_status["agents"] = agents
        base_status["messages"] = recent_messages
        base_status["timestamp"] = datetime.now().isoformat()
        
        return base_status
    
    async def broadcast_update(self, data):
        """Broadcast updates to all connected clients"""
        if self.websockets:
            await asyncio.gather(
                *[ws.send_json(data) for ws in self.websockets],
                return_exceptions=True
            )
    
    async def monitor_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                status = self.get_enhanced_status()
                status["latest_activity"] = f"System update - {len(self.system.agents)} agents active"
                await self.broadcast_update(status)
                await asyncio.sleep(2)  # Update every 2 seconds
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(5)
    
    async def start(self, host='localhost', port=8080):
        """Start the monitoring server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        
        print(f"🌐 Dashboard available at http://{host}:{port}")
        
        await site.start()
        
        # Start monitoring loop
        asyncio.create_task(self.monitor_loop())


async def run_with_monitoring():
    """Run the multi-agent system with monitoring dashboard"""
    from advanced_multiagent_system import MultiAgentSystem, demonstrate_system
    
    # Create system
    system = MultiAgentSystem()
    
    # Create and start monitor
    monitor = SystemMonitor(system)
    await monitor.start()
    
    # Run demonstration
    await demonstrate_system()
    
    # Keep running for monitoring
    print("\n📊 Monitoring dashboard running at http://localhost:8080")
    print("Press Ctrl+C to stop")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == "__main__":
    asyncio.run(run_with_monitoring())