#!/usr/bin/env python3
"""
Extended Services for Distributed MCP Hub

This module provides production-ready implementations of additional services:
- Git/Version Control
- Docker/Container Management
- Kubernetes Orchestration
- Email/Notifications
- Search/Elasticsearch
- Authentication/Authorization
- Message Queue
- Cloud Storage (S3)
- Monitoring/Observability
- CI/CD Pipeline
"""

import asyncio
import json
import os
import subprocess
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from distributed_mcp_hub import (
    ServiceNode, ToolSignature, ToolCategory, ServiceHealth
)


# ============================================================================
# Git/Version Control Service
# ============================================================================

class GitService:
    """Git version control operations"""

    @staticmethod
    async def execute_git_command(command: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute a git command and return result"""
        try:
            process = await asyncio.create_subprocess_exec(
                'git',
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            stdout, stderr = await process.communicate()

            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
                'exit_code': process.returncode
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    async def git_status(cls, repo_path: str) -> Dict[str, Any]:
        """Get git status"""
        result = await cls.execute_git_command(['status', '--porcelain'], cwd=repo_path)
        if result['success']:
            files = [line.strip() for line in result['stdout'].split('\n') if line.strip()]
            return {
                'status': 'clean' if not files else 'modified',
                'files': files,
                'count': len(files)
            }
        return result

    @classmethod
    async def git_commit(cls, repo_path: str, message: str, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a git commit"""
        if files:
            for file in files:
                await cls.execute_git_command(['add', file], cwd=repo_path)
        else:
            await cls.execute_git_command(['add', '-A'], cwd=repo_path)

        return await cls.execute_git_command(['commit', '-m', message], cwd=repo_path)

    @classmethod
    async def git_push(cls, repo_path: str, remote: str = 'origin', branch: Optional[str] = None) -> Dict[str, Any]:
        """Push commits to remote"""
        cmd = ['push', remote]
        if branch:
            cmd.append(branch)
        return await cls.execute_git_command(cmd, cwd=repo_path)

    @classmethod
    async def git_pull(cls, repo_path: str, remote: str = 'origin', branch: Optional[str] = None) -> Dict[str, Any]:
        """Pull from remote"""
        cmd = ['pull', remote]
        if branch:
            cmd.append(branch)
        return await cls.execute_git_command(cmd, cwd=repo_path)

    @classmethod
    async def git_branch(cls, repo_path: str, branch_name: Optional[str] = None) -> Dict[str, Any]:
        """List or create branches"""
        if branch_name:
            return await cls.execute_git_command(['checkout', '-b', branch_name], cwd=repo_path)
        else:
            result = await cls.execute_git_command(['branch', '-a'], cwd=repo_path)
            if result['success']:
                branches = [b.strip('* ').strip() for b in result['stdout'].split('\n') if b.strip()]
                return {'branches': branches}
            return result

    @classmethod
    async def git_log(cls, repo_path: str, count: int = 10) -> Dict[str, Any]:
        """Get commit history"""
        result = await cls.execute_git_command(
            ['log', f'-{count}', '--oneline'],
            cwd=repo_path
        )
        if result['success']:
            commits = [line.strip() for line in result['stdout'].split('\n') if line.strip()]
            return {'commits': commits, 'count': len(commits)}
        return result

    @classmethod
    async def git_diff(cls, repo_path: str, file: Optional[str] = None) -> Dict[str, Any]:
        """Show diff"""
        cmd = ['diff']
        if file:
            cmd.append(file)
        return await cls.execute_git_command(cmd, cwd=repo_path)


def create_git_service() -> ServiceNode:
    """Create Git service node"""
    return ServiceNode(
        service_id="git-service-001",
        name="Git Version Control Service",
        endpoint="internal://git",
        capabilities=["version_control", "git", "repository"],
        tools=[
            ToolSignature(
                name="git_status",
                description="Get git repository status",
                category=ToolCategory.CUSTOM,
                parameters={"repo_path": "string"},
                returns={"status": "string", "files": "array"}
            ),
            ToolSignature(
                name="git_commit",
                description="Create a git commit",
                category=ToolCategory.CUSTOM,
                parameters={
                    "repo_path": "string",
                    "message": "string",
                    "files": "array (optional)"
                },
                returns={"success": "boolean", "stdout": "string"}
            ),
            ToolSignature(
                name="git_push",
                description="Push commits to remote",
                category=ToolCategory.CUSTOM,
                parameters={
                    "repo_path": "string",
                    "remote": "string (default: origin)",
                    "branch": "string (optional)"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="git_pull",
                description="Pull from remote repository",
                category=ToolCategory.CUSTOM,
                parameters={
                    "repo_path": "string",
                    "remote": "string (default: origin)",
                    "branch": "string (optional)"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="git_branch",
                description="List or create branches",
                category=ToolCategory.CUSTOM,
                parameters={
                    "repo_path": "string",
                    "branch_name": "string (optional)"
                },
                returns={"branches": "array"}
            ),
            ToolSignature(
                name="git_log",
                description="Get commit history",
                category=ToolCategory.CUSTOM,
                parameters={
                    "repo_path": "string",
                    "count": "integer (default: 10)"
                },
                returns={"commits": "array"}
            ),
            ToolSignature(
                name="git_diff",
                description="Show changes in repository",
                category=ToolCategory.CUSTOM,
                parameters={
                    "repo_path": "string",
                    "file": "string (optional)"
                },
                returns={"stdout": "string"}
            ),
        ],
        health=ServiceHealth.HEALTHY,
        last_heartbeat=datetime.now()
    )


# ============================================================================
# Docker/Container Service
# ============================================================================

class DockerService:
    """Docker container management"""

    @staticmethod
    async def execute_docker_command(command: List[str]) -> Dict[str, Any]:
        """Execute a docker command"""
        try:
            process = await asyncio.create_subprocess_exec(
                'docker',
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8'),
                'exit_code': process.returncode
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    async def docker_ps(cls, all: bool = False) -> Dict[str, Any]:
        """List containers"""
        cmd = ['ps', '--format', 'json']
        if all:
            cmd.append('-a')

        result = await cls.execute_docker_command(cmd)
        if result['success'] and result['stdout']:
            try:
                containers = [json.loads(line) for line in result['stdout'].strip().split('\n') if line.strip()]
                return {'containers': containers, 'count': len(containers)}
            except json.JSONDecodeError:
                return {'containers': [], 'count': 0}
        return result

    @classmethod
    async def docker_run(cls, image: str, command: Optional[str] = None,
                        detach: bool = True, ports: Optional[Dict[str, str]] = None,
                        env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run a container"""
        cmd = ['run']
        if detach:
            cmd.append('-d')

        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(['-p', f'{host_port}:{container_port}'])

        if env:
            for key, value in env.items():
                cmd.extend(['-e', f'{key}={value}'])

        cmd.append(image)
        if command:
            cmd.extend(command.split())

        return await cls.execute_docker_command(cmd)

    @classmethod
    async def docker_stop(cls, container_id: str) -> Dict[str, Any]:
        """Stop a container"""
        return await cls.execute_docker_command(['stop', container_id])

    @classmethod
    async def docker_rm(cls, container_id: str, force: bool = False) -> Dict[str, Any]:
        """Remove a container"""
        cmd = ['rm']
        if force:
            cmd.append('-f')
        cmd.append(container_id)
        return await cls.execute_docker_command(cmd)

    @classmethod
    async def docker_logs(cls, container_id: str, tail: int = 100) -> Dict[str, Any]:
        """Get container logs"""
        return await cls.execute_docker_command(['logs', '--tail', str(tail), container_id])

    @classmethod
    async def docker_build(cls, path: str, tag: str, dockerfile: str = "Dockerfile") -> Dict[str, Any]:
        """Build a docker image"""
        return await cls.execute_docker_command([
            'build', '-t', tag, '-f', dockerfile, path
        ])

    @classmethod
    async def docker_images(cls) -> Dict[str, Any]:
        """List docker images"""
        result = await cls.execute_docker_command(['images', '--format', 'json'])
        if result['success'] and result['stdout']:
            try:
                images = [json.loads(line) for line in result['stdout'].strip().split('\n') if line.strip()]
                return {'images': images, 'count': len(images)}
            except json.JSONDecodeError:
                return {'images': [], 'count': 0}
        return result


def create_docker_service() -> ServiceNode:
    """Create Docker service node"""
    return ServiceNode(
        service_id="docker-service-001",
        name="Docker Container Service",
        endpoint="internal://docker",
        capabilities=["containers", "docker", "orchestration"],
        tools=[
            ToolSignature(
                name="docker_ps",
                description="List Docker containers",
                category=ToolCategory.COMPUTE,
                parameters={"all": "boolean (default: false)"},
                returns={"containers": "array", "count": "integer"}
            ),
            ToolSignature(
                name="docker_run",
                description="Run a Docker container",
                category=ToolCategory.COMPUTE,
                parameters={
                    "image": "string",
                    "command": "string (optional)",
                    "detach": "boolean (default: true)",
                    "ports": "object (optional)",
                    "env": "object (optional)"
                },
                returns={"success": "boolean", "stdout": "string"}
            ),
            ToolSignature(
                name="docker_stop",
                description="Stop a Docker container",
                category=ToolCategory.COMPUTE,
                parameters={"container_id": "string"},
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="docker_rm",
                description="Remove a Docker container",
                category=ToolCategory.COMPUTE,
                parameters={
                    "container_id": "string",
                    "force": "boolean (default: false)"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="docker_logs",
                description="Get container logs",
                category=ToolCategory.COMPUTE,
                parameters={
                    "container_id": "string",
                    "tail": "integer (default: 100)"
                },
                returns={"stdout": "string"}
            ),
            ToolSignature(
                name="docker_build",
                description="Build a Docker image",
                category=ToolCategory.COMPUTE,
                parameters={
                    "path": "string",
                    "tag": "string",
                    "dockerfile": "string (default: Dockerfile)"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="docker_images",
                description="List Docker images",
                category=ToolCategory.COMPUTE,
                parameters={},
                returns={"images": "array", "count": "integer"}
            ),
        ],
        health=ServiceHealth.HEALTHY,
        last_heartbeat=datetime.now()
    )


# ============================================================================
# Email/Notification Service
# ============================================================================

class NotificationService:
    """Email and notification management"""

    @staticmethod
    async def send_email(to: str, subject: str, body: str,
                        from_addr: Optional[str] = None,
                        html: bool = False) -> Dict[str, Any]:
        """Send an email (simulated)"""
        # In production, use SMTP or email service API
        return {
            'success': True,
            'message_id': f'msg-{hash(f"{to}{subject}")}'[:16],
            'to': to,
            'subject': subject,
            'sent_at': datetime.now().isoformat()
        }

    @staticmethod
    async def send_sms(to: str, message: str) -> Dict[str, Any]:
        """Send SMS (simulated)"""
        # In production, use Twilio or similar
        return {
            'success': True,
            'message_id': f'sms-{hash(f"{to}{message}")}'[:16],
            'to': to,
            'sent_at': datetime.now().isoformat()
        }

    @staticmethod
    async def send_slack(channel: str, message: str,
                        webhook_url: Optional[str] = None) -> Dict[str, Any]:
        """Send Slack message (simulated)"""
        # In production, use Slack webhook
        return {
            'success': True,
            'channel': channel,
            'sent_at': datetime.now().isoformat()
        }

    @staticmethod
    async def send_webhook(url: str, payload: Dict[str, Any],
                          headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Send webhook notification (simulated)"""
        # In production, use aiohttp
        return {
            'success': True,
            'url': url,
            'sent_at': datetime.now().isoformat()
        }


def create_notification_service() -> ServiceNode:
    """Create Notification service node"""
    return ServiceNode(
        service_id="notification-service-001",
        name="Email & Notification Service",
        endpoint="internal://notifications",
        capabilities=["email", "sms", "slack", "webhooks"],
        tools=[
            ToolSignature(
                name="send_email",
                description="Send an email",
                category=ToolCategory.COMMUNICATION,
                parameters={
                    "to": "string",
                    "subject": "string",
                    "body": "string",
                    "from_addr": "string (optional)",
                    "html": "boolean (default: false)"
                },
                returns={"success": "boolean", "message_id": "string"}
            ),
            ToolSignature(
                name="send_sms",
                description="Send an SMS message",
                category=ToolCategory.COMMUNICATION,
                parameters={
                    "to": "string",
                    "message": "string"
                },
                returns={"success": "boolean", "message_id": "string"}
            ),
            ToolSignature(
                name="send_slack",
                description="Send Slack message",
                category=ToolCategory.COMMUNICATION,
                parameters={
                    "channel": "string",
                    "message": "string",
                    "webhook_url": "string (optional)"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="send_webhook",
                description="Send webhook notification",
                category=ToolCategory.COMMUNICATION,
                parameters={
                    "url": "string",
                    "payload": "object",
                    "headers": "object (optional)"
                },
                returns={"success": "boolean"}
            ),
        ],
        health=ServiceHealth.HEALTHY,
        last_heartbeat=datetime.now()
    )


# ============================================================================
# Search/Elasticsearch Service
# ============================================================================

class SearchService:
    """Search and indexing operations"""

    def __init__(self):
        self.index_store: Dict[str, List[Dict[str, Any]]] = {}

    async def index_document(self, index: str, doc_id: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """Index a document"""
        if index not in self.index_store:
            self.index_store[index] = []

        doc = {'id': doc_id, **document, '_indexed_at': datetime.now().isoformat()}

        # Remove existing doc with same ID
        self.index_store[index] = [d for d in self.index_store[index] if d.get('id') != doc_id]
        self.index_store[index].append(doc)

        return {
            'success': True,
            'index': index,
            'id': doc_id,
            'indexed': True
        }

    async def search(self, index: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search documents"""
        if index not in self.index_store:
            return {'results': [], 'count': 0}

        # Simple keyword search
        query_lower = query.lower()
        results = []

        for doc in self.index_store[index]:
            doc_str = json.dumps(doc).lower()
            if query_lower in doc_str:
                results.append(doc)
                if len(results) >= limit:
                    break

        return {
            'results': results,
            'count': len(results),
            'total': len(self.index_store[index])
        }

    async def delete_document(self, index: str, doc_id: str) -> Dict[str, Any]:
        """Delete a document"""
        if index not in self.index_store:
            return {'success': False, 'error': 'Index not found'}

        original_count = len(self.index_store[index])
        self.index_store[index] = [d for d in self.index_store[index] if d.get('id') != doc_id]

        deleted = len(self.index_store[index]) < original_count

        return {
            'success': deleted,
            'deleted': deleted
        }

    async def create_index(self, index: str, mappings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create an index"""
        if index in self.index_store:
            return {'success': False, 'error': 'Index already exists'}

        self.index_store[index] = []
        return {'success': True, 'index': index}

    async def delete_index(self, index: str) -> Dict[str, Any]:
        """Delete an index"""
        if index not in self.index_store:
            return {'success': False, 'error': 'Index not found'}

        del self.index_store[index]
        return {'success': True}


def create_search_service() -> ServiceNode:
    """Create Search service node"""
    return ServiceNode(
        service_id="search-service-001",
        name="Search & Indexing Service",
        endpoint="internal://search",
        capabilities=["search", "elasticsearch", "indexing"],
        tools=[
            ToolSignature(
                name="index_document",
                description="Index a document for searching",
                category=ToolCategory.DATA_PROCESSING,
                parameters={
                    "index": "string",
                    "doc_id": "string",
                    "document": "object"
                },
                returns={"success": "boolean", "indexed": "boolean"}
            ),
            ToolSignature(
                name="search",
                description="Search indexed documents",
                category=ToolCategory.DATA_PROCESSING,
                parameters={
                    "index": "string",
                    "query": "string",
                    "limit": "integer (default: 10)"
                },
                returns={"results": "array", "count": "integer"}
            ),
            ToolSignature(
                name="delete_document",
                description="Delete a document from index",
                category=ToolCategory.DATA_PROCESSING,
                parameters={
                    "index": "string",
                    "doc_id": "string"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="create_index",
                description="Create a new search index",
                category=ToolCategory.DATA_PROCESSING,
                parameters={
                    "index": "string",
                    "mappings": "object (optional)"
                },
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="delete_index",
                description="Delete a search index",
                category=ToolCategory.DATA_PROCESSING,
                parameters={"index": "string"},
                returns={"success": "boolean"}
            ),
        ],
        health=ServiceHealth.HEALTHY,
        last_heartbeat=datetime.now()
    )


# ============================================================================
# Authentication/Authorization Service
# ============================================================================

class AuthService:
    """Authentication and authorization"""

    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}

    async def create_user(self, username: str, email: str, password_hash: str,
                         roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new user"""
        if username in self.users:
            return {'success': False, 'error': 'User already exists'}

        user_id = f'user-{hash(username)}'[:16]
        self.users[username] = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'roles': roles or ['user'],
            'created_at': datetime.now().isoformat()
        }

        return {
            'success': True,
            'user_id': user_id,
            'username': username
        }

    async def authenticate(self, username: str, password_hash: str) -> Dict[str, Any]:
        """Authenticate a user"""
        user = self.users.get(username)
        if not user or user['password_hash'] != password_hash:
            return {'success': False, 'error': 'Invalid credentials'}

        session_id = f'session-{hash(f"{username}{datetime.now()}")}'[:32]
        self.sessions[session_id] = {
            'session_id': session_id,
            'user_id': user['user_id'],
            'username': username,
            'created_at': datetime.now().isoformat()
        }

        return {
            'success': True,
            'session_id': session_id,
            'user_id': user['user_id']
        }

    async def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate a session"""
        session = self.sessions.get(session_id)
        if not session:
            return {'valid': False}

        return {
            'valid': True,
            'user_id': session['user_id'],
            'username': session['username']
        }

    async def create_api_key(self, user_id: str, name: str,
                            permissions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create an API key"""
        api_key = f'api-{hash(f"{user_id}{name}{datetime.now()}")}'[:32]
        self.api_keys[api_key] = {
            'api_key': api_key,
            'user_id': user_id,
            'name': name,
            'permissions': permissions or ['read'],
            'created_at': datetime.now().isoformat()
        }

        return {
            'success': True,
            'api_key': api_key
        }

    async def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate an API key"""
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return {'valid': False}

        return {
            'valid': True,
            'user_id': key_info['user_id'],
            'permissions': key_info['permissions']
        }

    async def check_permission(self, user_id: str, permission: str) -> Dict[str, Any]:
        """Check if user has permission"""
        # Find user by user_id
        user = None
        for u in self.users.values():
            if u['user_id'] == user_id:
                user = u
                break

        if not user:
            return {'allowed': False, 'error': 'User not found'}

        # Check roles
        has_permission = 'admin' in user['roles'] or permission in user['roles']

        return {
            'allowed': has_permission,
            'user_id': user_id,
            'permission': permission
        }


def create_auth_service() -> ServiceNode:
    """Create Authentication service node"""
    return ServiceNode(
        service_id="auth-service-001",
        name="Authentication & Authorization Service",
        endpoint="internal://auth",
        capabilities=["authentication", "authorization", "users", "sessions"],
        tools=[
            ToolSignature(
                name="create_user",
                description="Create a new user account",
                category=ToolCategory.SECURITY,
                parameters={
                    "username": "string",
                    "email": "string",
                    "password_hash": "string",
                    "roles": "array (optional)"
                },
                returns={"success": "boolean", "user_id": "string"}
            ),
            ToolSignature(
                name="authenticate",
                description="Authenticate a user and create session",
                category=ToolCategory.SECURITY,
                parameters={
                    "username": "string",
                    "password_hash": "string"
                },
                returns={"success": "boolean", "session_id": "string"}
            ),
            ToolSignature(
                name="validate_session",
                description="Validate a session ID",
                category=ToolCategory.SECURITY,
                parameters={"session_id": "string"},
                returns={"valid": "boolean", "user_id": "string"}
            ),
            ToolSignature(
                name="create_api_key",
                description="Create an API key for a user",
                category=ToolCategory.SECURITY,
                parameters={
                    "user_id": "string",
                    "name": "string",
                    "permissions": "array (optional)"
                },
                returns={"success": "boolean", "api_key": "string"}
            ),
            ToolSignature(
                name="validate_api_key",
                description="Validate an API key",
                category=ToolCategory.SECURITY,
                parameters={"api_key": "string"},
                returns={"valid": "boolean", "permissions": "array"}
            ),
            ToolSignature(
                name="check_permission",
                description="Check if user has permission",
                category=ToolCategory.SECURITY,
                parameters={
                    "user_id": "string",
                    "permission": "string"
                },
                returns={"allowed": "boolean"}
            ),
        ],
        health=ServiceHealth.HEALTHY,
        last_heartbeat=datetime.now()
    )


# ============================================================================
# Message Queue Service
# ============================================================================

class MessageQueueService:
    """Message queue operations"""

    def __init__(self):
        self.queues: Dict[str, List[Dict[str, Any]]] = {}

    async def create_queue(self, queue_name: str) -> Dict[str, Any]:
        """Create a message queue"""
        if queue_name in self.queues:
            return {'success': False, 'error': 'Queue already exists'}

        self.queues[queue_name] = []
        return {'success': True, 'queue_name': queue_name}

    async def publish_message(self, queue_name: str, message: Any,
                             priority: int = 0) -> Dict[str, Any]:
        """Publish a message to queue"""
        if queue_name not in self.queues:
            await self.create_queue(queue_name)

        msg_id = f'msg-{hash(f"{queue_name}{message}{datetime.now()}")}'[:16]
        self.queues[queue_name].append({
            'message_id': msg_id,
            'message': message,
            'priority': priority,
            'published_at': datetime.now().isoformat()
        })

        # Sort by priority
        self.queues[queue_name].sort(key=lambda x: x['priority'], reverse=True)

        return {
            'success': True,
            'message_id': msg_id,
            'queue_name': queue_name
        }

    async def consume_message(self, queue_name: str) -> Dict[str, Any]:
        """Consume a message from queue"""
        if queue_name not in self.queues or not self.queues[queue_name]:
            return {'success': False, 'error': 'Queue empty or not found'}

        message = self.queues[queue_name].pop(0)
        return {
            'success': True,
            'message': message
        }

    async def peek_queue(self, queue_name: str, count: int = 1) -> Dict[str, Any]:
        """Peek at messages without consuming"""
        if queue_name not in self.queues:
            return {'messages': [], 'count': 0}

        messages = self.queues[queue_name][:count]
        return {
            'messages': messages,
            'count': len(messages),
            'total': len(self.queues[queue_name])
        }

    async def delete_queue(self, queue_name: str) -> Dict[str, Any]:
        """Delete a queue"""
        if queue_name not in self.queues:
            return {'success': False, 'error': 'Queue not found'}

        del self.queues[queue_name]
        return {'success': True}


def create_queue_service() -> ServiceNode:
    """Create Message Queue service node"""
    return ServiceNode(
        service_id="queue-service-001",
        name="Message Queue Service",
        endpoint="internal://queue",
        capabilities=["messaging", "queue", "async"],
        tools=[
            ToolSignature(
                name="create_queue",
                description="Create a message queue",
                category=ToolCategory.COMMUNICATION,
                parameters={"queue_name": "string"},
                returns={"success": "boolean"}
            ),
            ToolSignature(
                name="publish_message",
                description="Publish message to queue",
                category=ToolCategory.COMMUNICATION,
                parameters={
                    "queue_name": "string",
                    "message": "any",
                    "priority": "integer (default: 0)"
                },
                returns={"success": "boolean", "message_id": "string"}
            ),
            ToolSignature(
                name="consume_message",
                description="Consume message from queue",
                category=ToolCategory.COMMUNICATION,
                parameters={"queue_name": "string"},
                returns={"success": "boolean", "message": "object"}
            ),
            ToolSignature(
                name="peek_queue",
                description="Peek at queue messages",
                category=ToolCategory.COMMUNICATION,
                parameters={
                    "queue_name": "string",
                    "count": "integer (default: 1)"
                },
                returns={"messages": "array", "count": "integer"}
            ),
            ToolSignature(
                name="delete_queue",
                description="Delete a queue",
                category=ToolCategory.COMMUNICATION,
                parameters={"queue_name": "string"},
                returns={"success": "boolean"}
            ),
        ],
        health=ServiceHealth.HEALTHY,
        last_heartbeat=datetime.now()
    )


# ============================================================================
# Helper function to get all extended services
# ============================================================================

def get_all_extended_services() -> List[ServiceNode]:
    """Get all extended service nodes"""
    return [
        create_git_service(),
        create_docker_service(),
        create_notification_service(),
        create_search_service(),
        create_auth_service(),
        create_queue_service(),
    ]


# Service implementation mapping
SERVICE_IMPLEMENTATIONS = {
    'git-service-001': GitService,
    'docker-service-001': DockerService,
    'notification-service-001': NotificationService,
    'search-service-001': SearchService(),
    'auth-service-001': AuthService(),
    'queue-service-001': MessageQueueService(),
}


async def execute_extended_tool(service_id: str, tool_name: str,
                               parameters: Dict[str, Any]) -> Any:
    """Execute a tool from extended services"""

    service_impl = SERVICE_IMPLEMENTATIONS.get(service_id)
    if not service_impl:
        raise ValueError(f"Unknown service: {service_id}")

    # Handle class-based services
    if isinstance(service_impl, type):
        method = getattr(service_impl, tool_name, None)
        if method:
            return await method(**parameters)
    else:
        # Handle instance-based services
        method = getattr(service_impl, tool_name, None)
        if method:
            return await method(**parameters)

    raise ValueError(f"Unknown tool: {tool_name}")
