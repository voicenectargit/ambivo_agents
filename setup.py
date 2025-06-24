#!/usr/bin/env python3
"""
Setup script for Ambivo Agents

Author: Hemant Gosain 'Sunny'
Company: Ambivo
Email: sgosain@ambivo.com
License: MIT
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ambivo-agents",
    version="1.0.0",
    author="Hemant Gosain 'Sunny'",
    author_email="sgosain@ambivo.com",
    description="A minimalistic multi-agent system for AI-powered automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ambivo-agents",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ambivo-agents/issues",
        "Documentation": "https://docs.ambivo.com/agents",
        "Source Code": "https://github.com/yourusername/ambivo-agents",
        "Company": "https://www.ambivo.com",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",

    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ambivo-agents=ambivo_agents.cli:main",
            "ambivo-server=ambivo_agents.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ambivo_agents": ["*.yaml", "*.yml"],
    },

    # Additional metadata
    maintainer="Hemant Gosain 'Sunny'",
    maintainer_email="sgosain@ambivo.com",
    license="MIT",
    keywords=[
        "ai", "agents", "automation", "multiagent", "llm", "openai", "anthropic",
        "web-scraping", "media-processing", "knowledge-base", "vector-database",
        "docker", "redis", "qdrant", "langchain", "llamaindex", "ffmpeg",
    ],
)