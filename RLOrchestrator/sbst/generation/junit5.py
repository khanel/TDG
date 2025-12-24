from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..discovery.models import ProjectUnderTest
from .models import GeneratedTest


@dataclass(frozen=True)
class GenerationConfig:
    max_tests_per_candidate: int = 1
    max_actions_per_test: int = 5
    package_strategy: str = "match_target"  # match_target | fixed
    fixed_package: str = "generated.sbst"


def generate_junit5_tests(
    *,
    candidate: Dict[str, List[int]],
    project: ProjectUnderTest,
    candidate_digest: str,
    targets: Optional[Sequence[str]] = None,
    cfg: Optional[GenerationConfig] = None,
) -> List[GeneratedTest]:
    """Generate deterministic JUnit 5 tests.

    MVP+ approach: reflection-based calls to public constructors and public methods.
    We invoke:
    - 0-arg methods always
    - 1–2 arg methods when parameters are simple types (primitives/wrappers/String)
    Calls are wrapped in assertDoesNotThrow to avoid flakiness.

    Determinism: controlled only by (candidate genes, project targets list, config).
    """

    cfg = cfg or GenerationConfig()

    available_targets = list(targets) if targets else list(project.target_classes)
    if not available_targets:
        return [
            _trivial_test(
                package=_select_package(cfg, ""),
                class_name=f"GeneratedSBST_{candidate_digest}_00",
                note="No target classes discovered.",
            )
        ]

    genes = list(candidate.get("genes") or [])
    if not genes:
        genes = [0]

    tests: List[GeneratedTest] = []
    for test_index in range(max(1, int(cfg.max_tests_per_candidate))):
        g0 = genes[(0 + test_index) % len(genes)]
        target = available_targets[int(g0) % len(available_targets)]
        package = _select_package(cfg, target)
        class_name = f"GeneratedSBST_{candidate_digest}_{test_index:02d}"

        # Build action tuples from genes: (method_selector, arg1_gene, arg2_gene).
        actions: List[Tuple[int, int, int]] = []
        for i in range(int(cfg.max_actions_per_test)):
            base = 1 + test_index + (i * 3)
            sel = int(genes[(base + 0) % len(genes)])
            a1 = int(genes[(base + 1) % len(genes)])
            a2 = int(genes[(base + 2) % len(genes)])
            actions.append((sel, a1, a2))

        tests.append(_build_reflection_test(package, class_name, target, actions))

    return tests


def generated_tests_digest(tests: Sequence[GeneratedTest]) -> str:
    h = hashlib.sha256()
    for t in tests:
        h.update(t.package.encode("utf-8"))
        h.update(b"\0")
        h.update(t.class_name.encode("utf-8"))
        h.update(b"\0")
        h.update(t.source.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _select_package(cfg: GenerationConfig, target_fqn: str) -> str:
    if cfg.package_strategy == "fixed":
        return cfg.fixed_package
    # match_target
    if "." in target_fqn:
        return target_fqn.rsplit(".", 1)[0]
    return cfg.fixed_package


def _build_reflection_test(package: str, class_name: str, target_fqn: str, actions: List[Tuple[int, int, int]]) -> GeneratedTest:
    # Reflection calls public members only; avoids illegal access.
    # This generator supports composition/stateful sequences by maintaining a small
    # object pool (reference slots) and a value pool (boxed primitives/strings).

    body_lines: List[str] = []
    body_lines.append(f"Class<?> clazz = Class.forName(\"{target_fqn}\");")
    body_lines.append("final int OBJ_SLOTS = 6;")
    body_lines.append("final int VAL_SLOTS = 6;")
    body_lines.append("Object[] objs = new Object[OBJ_SLOTS];")
    body_lines.append("Object[] vals = new Object[VAL_SLOTS];")
    body_lines.append("Object instance = null;")
    body_lines.append("try {")
    body_lines.append("    instance = tryConstruct(clazz, 0, 1, objs, vals);")
    body_lines.append("} catch (Throwable ignored) {")
    body_lines.append("    // No public constructor worked; we'll only call static methods.")
    body_lines.append("}")
    body_lines.append("objs[0] = instance;")
    body_lines.append("// Seed a few common JDK objects to make composition feasible.")
    body_lines.append("if (OBJ_SLOTS > 1) objs[1] = new java.util.ArrayList<>();")
    body_lines.append("if (OBJ_SLOTS > 2) objs[2] = new java.util.HashMap<>();")
    body_lines.append("if (OBJ_SLOTS > 3) objs[3] = new java.util.HashSet<>();")
    body_lines.append("if (OBJ_SLOTS > 4) objs[4] = new StringBuilder();")
    body_lines.append("if (VAL_SLOTS > 0) vals[0] = Integer.valueOf(0);")
    body_lines.append("if (VAL_SLOTS > 1) vals[1] = Boolean.TRUE;")
    body_lines.append("if (VAL_SLOTS > 2) vals[2] = \"a\";")

    for i, (sel, a1, a2) in enumerate(actions):
        body_lines.append(f"final int recvSlot{i} = Math.floorMod({int(sel)}, objs.length);")
        body_lines.append(f"Object recv{i} = objs[recvSlot{i}];")
        body_lines.append(f"Class<?> recvClass{i} = (recv{i} != null) ? recv{i}.getClass() : clazz;")
        body_lines.append(f"java.util.List<java.lang.reflect.Method> callables{i} = collectCallables(recvClass{i});")
        body_lines.append(f"if (!callables{i}.isEmpty()) {{")
        body_lines.append(f"    int midx{i} = Math.floorMod({int(a1)}, callables{i}.size());")
        body_lines.append(f"    java.lang.reflect.Method m{i} = callables{i}.get(midx{i});")
        body_lines.append(f"    int pc{i} = m{i}.getParameterCount();")
        body_lines.append(f"    Object[] args{i} = buildArgs(m{i}, {int(a1)}, {int(a2)}, objs, vals);")
        body_lines.append(f"    if ((pc{i} == 0) || (args{i} != null && args{i}.length == pc{i})) {{")
        body_lines.append(f"        boolean isStatic{i} = java.lang.reflect.Modifier.isStatic(m{i}.getModifiers());")
        body_lines.append(f"        if (isStatic{i}) {{")
        body_lines.append(f"            try {{")
        body_lines.append(f"                Object r{i} = m{i}.invoke(null, args{i});")
        body_lines.append(f"                storeReturn(m{i}.getReturnType(), r{i}, {int(a1)}, {int(a2)}, objs, vals);")
        body_lines.append(f"            }} catch (Throwable ignored) {{ }}")
        body_lines.append(f"        }} else if (recv{i} != null) {{")
        body_lines.append(f"            try {{")
        body_lines.append(f"                Object r{i} = m{i}.invoke(recv{i}, args{i});")
        body_lines.append(f"                storeReturn(m{i}.getReturnType(), r{i}, {int(a1)}, {int(a2)}, objs, vals);")
        body_lines.append(f"            }} catch (Throwable ignored) {{ }}")
        body_lines.append(f"        }} else if (instance != null) {{")
        body_lines.append(f"            // Fallback to the primary target instance when receiver slot is null.")
        body_lines.append(f"            try {{")
        body_lines.append(f"                Object r{i} = m{i}.invoke(instance, args{i});")
        body_lines.append(f"                storeReturn(m{i}.getReturnType(), r{i}, {int(a1)}, {int(a2)}, objs, vals);")
        body_lines.append(f"            }} catch (Throwable ignored) {{ }}")
        body_lines.append(f"        }}")
        body_lines.append(f"    }}")
        body_lines.append(f"}}")

    body = "\n        ".join(body_lines)

    source = f"""package {package};

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class {class_name} {{

    private static boolean isConfigLikeName(String name) {{
        if (name == null) return false;
        // Common mutators/configurators that help initialize composed objects.
        return name.startsWith("set") || name.startsWith("add") || name.startsWith("put")
            || name.startsWith("enable") || name.startsWith("disable")
            || name.startsWith("init") || name.startsWith("register")
            || name.startsWith("configure") || name.startsWith("with")
            || name.startsWith("append") || name.startsWith("insert")
            || name.startsWith("push") || name.startsWith("offer");
    }}

    private static boolean isCollectionReceiver(Class<?> clazz) {{
        if (clazz == null) return false;
        try {{
            return java.util.Map.class.isAssignableFrom(clazz)
                || java.util.Collection.class.isAssignableFrom(clazz)
                || clazz == StringBuilder.class;
        }} catch (Throwable ignored) {{
            return false;
        }}
    }}

    private static int methodPriority(Class<?> receiver, java.lang.reflect.Method m) {{
        // Lower is better.
        String n = (m != null) ? m.getName() : null;
        if (n == null) return 100;

        // Strong bias: when the receiver is a collection-like JDK type, prefer
        // methods that mutate/populate it first (helps composition quickly).
        if (isCollectionReceiver(receiver)) {{
            if (n.equals("put") || n.equals("putAll") || n.equals("add") || n.equals("addAll")
                || n.equals("offer") || n.equals("push") || n.equals("append")
                || n.equals("insert") || n.equals("set") || n.equals("replace")) {{
                return 0;
            }}
            if (n.equals("remove") || n.equals("clear") || n.equals("poll") || n.equals("pop")) {{
                return 2;
            }}
            if (n.equals("get") || n.equals("contains") || n.equals("containsKey") || n.equals("containsValue")
                || n.equals("size") || n.equals("isEmpty") || n.equals("toString")) {{
                return 3;
            }}
            if (isConfigLikeName(n)) {{
                return 1;
            }}
            return 4;
        }}

        // General case: config-like methods first.
        if (isConfigLikeName(n)) return 1;
        return 2;
    }}

    private static java.util.List<java.lang.reflect.Method> collectCallables(Class<?> clazz) {{
        java.lang.reflect.Method[] methods = clazz.getMethods();
        java.util.List<java.lang.reflect.Method> callables = new java.util.ArrayList<>();
        for (java.lang.reflect.Method m : methods) {{
            // Skip Object methods to reduce noise.
            if (m.getDeclaringClass() == Object.class) continue;
            int pc = m.getParameterCount();
            if (pc > 2) continue;

            callables.add(m);
        }}

        java.util.Comparator<java.lang.reflect.Method> cmp = (a, b) -> {{
            int pa = methodPriority(clazz, a);
            int pb = methodPriority(clazz, b);
            if (pa != pb) return Integer.compare(pa, pb);
            return a.toString().compareTo(b.toString());
        }};
        java.util.Collections.sort(callables, cmp);
        return callables;
    }}

    private static boolean isSimpleType(Class<?> t) {{
        if (t.isPrimitive()) {{
            return t == int.class || t == long.class || t == float.class || t == double.class || t == boolean.class || t == char.class;
        }}
        return t == Integer.class || t == Long.class || t == Float.class || t == Double.class || t == Boolean.class || t == Character.class || t == String.class;
    }}

    private static boolean isJdkCollectionLike(Class<?> t) {{
        return t == java.util.List.class || t == java.util.Set.class || t == java.util.Map.class || t == java.util.Collection.class;
    }}

    private static Object defaultJdkInstance(Class<?> t) {{
        try {{
            if (t == java.util.List.class || t == java.util.Collection.class || t.isAssignableFrom(java.util.ArrayList.class)) {{
                return new java.util.ArrayList<>();
            }}
            if (t == java.util.Set.class || t.isAssignableFrom(java.util.HashSet.class)) {{
                return new java.util.HashSet<>();
            }}
            if (t == java.util.Map.class || t.isAssignableFrom(java.util.HashMap.class)) {{
                return new java.util.HashMap<>();
            }}
            if (t == StringBuilder.class) {{
                return new StringBuilder();
            }}
        }} catch (Throwable ignored) {{ }}
        return null;
    }}

    private static Object coerceSimple(Class<?> t, int gene) {{
        // Allow nulls for reference types (useful for null-check branches).
        // Keep deterministic and conservative: ~1/8 of the time.
        if (!t.isPrimitive()) {{
            if (Math.floorMod(gene, 8) == 0) {{
                return null;
            }}
        }}
        if (t == int.class || t == Integer.class) {{
            return Integer.valueOf(gene);
        }}
        if (t == long.class || t == Long.class) {{
            return Long.valueOf((long) gene);
        }}
        if (t == float.class || t == Float.class) {{
            return Float.valueOf(((float) gene) / 10.0f);
        }}
        if (t == double.class || t == Double.class) {{
            return Double.valueOf(((double) gene) / 10.0);
        }}
        if (t == boolean.class || t == Boolean.class) {{
            return Boolean.valueOf((gene & 1) == 0);
        }}
        if (t == char.class || t == Character.class) {{
            char c = (char) ('a' + Math.floorMod(gene, 26));
            return Character.valueOf(c);
        }}
        if (t == String.class) {{
            int len = 1 + Math.floorMod(gene, 8);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < len; i++) {{
                char c = (char) ('a' + Math.floorMod(gene + i, 26));
                sb.append(c);
            }}
            return sb.toString();
        }}
        return null;
    }}

    private static Object fromValuePool(Class<?> t, int gene, Object[] vals) {{
        if (vals == null || vals.length == 0) return null;
        Object v = vals[Math.floorMod(gene, vals.length)];
        if (v == null) return null;
        // Reflection accepts boxed for primitives.
        if (t == int.class || t == Integer.class) return (v instanceof Integer) ? v : null;
        if (t == long.class || t == Long.class) return (v instanceof Long) ? v : null;
        if (t == float.class || t == Float.class) return (v instanceof Float) ? v : null;
        if (t == double.class || t == Double.class) return (v instanceof Double) ? v : null;
        if (t == boolean.class || t == Boolean.class) return (v instanceof Boolean) ? v : null;
        if (t == char.class || t == Character.class) return (v instanceof Character) ? v : null;
        if (t == String.class) return (v instanceof String) ? v : null;
        if (!t.isPrimitive() && t.isInstance(v)) return v;
        return null;
    }}

    private static Object tryConstruct(Class<?> t, int g1, int g2, Object[] objs, Object[] vals) {{
        try {{
            if (t.isPrimitive()) return null;
            int mods = t.getModifiers();
            if (java.lang.reflect.Modifier.isInterface(mods) || java.lang.reflect.Modifier.isAbstract(mods) || isJdkCollectionLike(t)) {{
                Object dk = defaultJdkInstance(t);
                if (dk != null) return dk;
                return null;
            }}

            // Prefer no-arg.
            try {{
                return t.getConstructor().newInstance();
            }} catch (Throwable ignored) {{ }}

            java.lang.reflect.Constructor<?>[] ctors = t.getConstructors();
            java.util.Arrays.sort(ctors, (a, b) -> a.toString().compareTo(b.toString()));
            for (java.lang.reflect.Constructor<?> c : ctors) {{
                int pc = c.getParameterCount();
                if (pc == 0) continue;
                if (pc > 2) continue;
                Class<?>[] pts = c.getParameterTypes();
                Object[] args = new Object[pc];
                boolean ok = true;
                for (int i = 0; i < pc; i++) {{
                    int gene = (i == 0) ? g1 : g2;
                    Object fromPool = fromValuePool(pts[i], gene, vals);
                    Object v = (fromPool != null) ? fromPool : coerceSimple(pts[i], gene);
                    if (v == null && pts[i].isPrimitive()) {{ ok = false; break; }}
                    args[i] = v;
                }}
                if (!ok) continue;
                try {{
                    return c.newInstance(args);
                }} catch (Throwable ignored) {{ }}
            }}
        }} catch (Throwable ignored) {{ }}
        return null;
    }}

    private static Object buildArg(Class<?> t, int gene, Object[] objs, Object[] vals) {{
        // Nulls for reference types.
        if (!t.isPrimitive() && Math.floorMod(gene, 8) == 0) {{
            return null;
        }}

        Object pooled = fromValuePool(t, gene, vals);
        if (pooled != null) return pooled;

        if (isSimpleType(t)) {{
            return coerceSimple(t, gene);
        }}

        // Try object pool.
        if (objs != null && objs.length > 0) {{
            Object cand = objs[Math.floorMod(gene, objs.length)];
            if (cand != null && t.isInstance(cand)) {{
                return cand;
            }}
        }}

        // Try constructing a new instance.
        Object dk = defaultJdkInstance(t);
        if (dk != null) return dk;
        return tryConstruct(t, gene, gene + 1, objs, vals);
    }}

    private static Object[] buildArgs(java.lang.reflect.Method m, int g1, int g2, Object[] objs, Object[] vals) {{
        int pc = m.getParameterCount();
        if (pc == 0) {{
            return new Object[0];
        }}
        if (pc > 2) {{
            return null;
        }}
        Class<?>[] pts = m.getParameterTypes();
        Object[] out = new Object[pc];
        if (pc >= 1) {{
            Object v1 = buildArg(pts[0], g1, objs, vals);
            if (v1 == null && pts[0].isPrimitive()) return null;
            out[0] = v1;
        }}
        if (pc == 2) {{
            Object v2 = buildArg(pts[1], g2, objs, vals);
            if (v2 == null && pts[1].isPrimitive()) return null;
            out[1] = v2;
        }}
        return out;
    }}

    private static void storeReturn(Class<?> rt, Object r, int g1, int g2, Object[] objs, Object[] vals) {{
        if (rt == void.class) return;
        if (r == null) return;
        int vslot = (vals == null || vals.length == 0) ? 0 : Math.floorMod(g1, vals.length);
        int oslot = (objs == null || objs.length == 0) ? 0 : Math.floorMod(g2, objs.length);

        // Store all returns in vals to enable reuse; also store non-simple references in objs.
        if (vals != null && vals.length > 0) {{
            vals[vslot] = r;
        }}
        if (!isSimpleType(rt) && objs != null && objs.length > 0) {{
            objs[oslot] = r;
        }}
    }}

    @Test
    public void test_{class_name}() {{
        assertDoesNotThrow(() -> {{
            {body}
        }});
    }}
}}
"""

    return GeneratedTest(package=package, class_name=class_name, source=source)


def _trivial_test(*, package: str, class_name: str, note: str) -> GeneratedTest:
    source = f"""package {package};

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class {class_name} {{

    @Test
    public void test_{class_name}() {{
        assertDoesNotThrow(() -> {{
            // {note}
        }});
    }}
}}
"""
    return GeneratedTest(package=package, class_name=class_name, source=source)
