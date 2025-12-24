from __future__ import annotations

from RLOrchestrator.sbst.discovery.models import ProjectUnderTest
from RLOrchestrator.sbst.generation import GenerationConfig, generate_junit5_tests, generated_tests_digest


def test_generation_is_deterministic_for_same_candidate_and_project():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=["/tmp/sut/src/main/java"],
        test_roots=["/tmp/sut/src/test/java"],
        target_classes=["com.acme.Foo", "com.acme.Bar"],
        hierarchy={},
    )

    candidate = {"genes": [1, 2, 3, 4, 5]}
    cfg = GenerationConfig(max_tests_per_candidate=2, max_actions_per_test=3)

    t1 = generate_junit5_tests(candidate=candidate, project=project, candidate_digest="abc123", cfg=cfg)
    t2 = generate_junit5_tests(candidate=candidate, project=project, candidate_digest="abc123", cfg=cfg)

    assert generated_tests_digest(t1) == generated_tests_digest(t2)
    assert [x.source for x in t1] == [x.source for x in t2]


def test_generation_uses_target_package_by_default():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=[],
        test_roots=[],
        target_classes=["com.acme.Foo"],
        hierarchy={},
    )
    candidate = {"genes": [0]}

    tests = generate_junit5_tests(candidate=candidate, project=project, candidate_digest="d", cfg=GenerationConfig())
    assert tests[0].package == "com.acme"
    assert "package com.acme;" in tests[0].source


def test_generation_emits_float_support_helpers():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=[],
        test_roots=[],
        target_classes=["com.acme.Foo"],
        hierarchy={},
    )
    tests = generate_junit5_tests(candidate={"genes": [0, 1, 2]}, project=project, candidate_digest="f", cfg=GenerationConfig())
    src = tests[0].source
    assert "float.class" in src
    assert "Float.class" in src
    assert "/ 10.0f" in src


def test_generation_emits_null_support_for_reference_types():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=[],
        test_roots=[],
        target_classes=["com.acme.Foo"],
        hierarchy={},
    )
    tests = generate_junit5_tests(candidate={"genes": [0, 1, 2]}, project=project, candidate_digest="n", cfg=GenerationConfig())
    src = tests[0].source
    assert "Allow nulls for reference types" in src
    assert "Math.floorMod(gene, 8) == 0" in src


def test_generation_emits_object_and_value_slots_for_composition():
    project = ProjectUnderTest(
        root_path="/tmp/sut",
        build_tool="maven",
        source_roots=[],
        test_roots=[],
        target_classes=["com.acme.Foo"],
        hierarchy={},
    )
    tests = generate_junit5_tests(candidate={"genes": [1, 2, 3]}, project=project, candidate_digest="s", cfg=GenerationConfig())
    src = tests[0].source
    assert "Object[] objs" in src
    assert "Object[] vals" in src
    assert "storeReturn" in src
    assert "isConfigLikeName" in src
    assert "methodPriority" in src
