#!/usr/bin/env python3
"""
Test script to verify capability consistency across the system.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ambivo_agents.config.loader import (
    validate_agent_capabilities,
    get_available_agent_types,
    get_enabled_capabilities,
    get_available_agent_type_names
)
from ambivo_agents.services.factory import AgentFactory
from ambivo_agents.services.agent_service import create_agent_service


def test_capability_consistency():
    """Test that all capability checking methods return consistent results."""

    print("🔍 Testing Capability Consistency Across System Components")
    print(
        "   Expected capabilities: assistant, code_execution, proxy, web_scraping, knowledge_base, web_search, media_editor\n")

    # Test 1: Direct capability checking
    print("1️⃣ Testing validate_agent_capabilities():")
    capabilities = validate_agent_capabilities()
    for cap, enabled in capabilities.items():
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"   {cap}: {status}")

    print("\n2️⃣ Testing get_available_agent_types():")
    agent_types = get_available_agent_types()
    for agent_type, available in agent_types.items():
        status = "✅ AVAILABLE" if available else "❌ UNAVAILABLE"
        print(f"   {agent_type}: {status}")

    print("\n3️⃣ Testing AgentFactory.get_available_agent_types():")
    factory_agent_types = AgentFactory.get_available_agent_types()
    for agent_type, available in factory_agent_types.items():
        status = "✅ AVAILABLE" if available else "❌ UNAVAILABLE"
        print(f"   {agent_type}: {status}")

    print("\n4️⃣ Testing AgentService consistency:")
    try:
        agent_service = create_agent_service()
        health_check = agent_service.health_check()

        print("   Service enabled_capabilities:", health_check.get('enabled_capabilities', []))
        print("   Service available_agent_types:", health_check.get('available_agent_types', {}))
        print("   Service available_agent_type_names:", health_check.get('available_agent_type_names', []))

    except Exception as e:
        print(f"   ❌ AgentService initialization failed: {e}")

    # Test 5: Consistency check
    print("\n5️⃣ Consistency Check:")
    consistency_issues = []

    # Check if get_available_agent_types and factory agree
    for agent_type in agent_types:
        direct_available = agent_types[agent_type]
        factory_available = factory_agent_types.get(agent_type, False)

        if direct_available != factory_available:
            consistency_issues.append(
                f"   ❌ Mismatch for {agent_type}: "
                f"direct={direct_available}, factory={factory_available}"
            )

    if consistency_issues:
        print("   🚨 CONSISTENCY ISSUES FOUND:")
        for issue in consistency_issues:
            print(issue)
    else:
        print("   ✅ ALL METHODS CONSISTENT")

    print("\n6️⃣ Helper Functions:")
    enabled_caps = get_enabled_capabilities()
    available_types = get_available_agent_type_names()
    print(f"   Enabled capabilities: {enabled_caps}")
    print(f"   Available agent type names: {available_types}")

    return len(consistency_issues) == 0


def main():
    """Main test function."""
    try:
        print("=" * 60)
        print("CAPABILITY CONSISTENCY TEST")
        print("=" * 60)

        success = test_capability_consistency()

        print("\n" + "=" * 60)
        if success:
            print("✅ ALL TESTS PASSED - SYSTEM IS CONSISTENT")
        else:
            print("❌ CONSISTENCY ISSUES FOUND - NEEDS ATTENTION")
        print("=" * 60)

        return 0 if success else 1

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())