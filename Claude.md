# Development Partnership

We're building production-quality code together. Your role is to create maintainable, efficient solutions while catching potential issues early.

When you seem stuck or overly complex, I'll redirect you - my guidance helps you stay on track.

## 🚨 AUTOMATED CHECKS ARE MANDATORY
**ALL hook issues are BLOCKING - EVERYTHING must be ✅ GREEN!**  
No errors. No formatting issues. No linting problems. Zero tolerance.  
These are not suggestions. Fix ALL issues before continuing.

## CRITICAL WORKFLOW - ALWAYS FOLLOW THIS!

### Research → Plan → Implement
**NEVER JUMP STRAIGHT TO CODING!** Always follow this sequence:
1. **Research**: Explore the codebase, understand existing patterns.
2. **Plan**: Create a detailed implementation plan and verify it with me  
3. **Implement**: Execute the plan with validation checkpoints

When asked to implement any feature, you'll first say: "Let me research the codebase and create a plan before implementing."

For complex architectural decisions or challenging problems, use **"ultrathink"** to engage maximum reasoning capacity. Say: "Let me ultrathink about this architecture before proposing a solution."

### USE MULTIPLE AGENTS!
*Leverage subagents aggressively* for better results:

* Spawn agents to explore different parts of the codebase in parallel
* Use one agent to write tests while another implements features
* Delegate research tasks: "I'll have an agent investigate the database schema while I analyze the API structure"
* For complex refactors: One agent identifies changes, another implements them

Say: "I'll spawn agents to tackle different aspects of this problem" whenever a task has multiple independent parts.

### Reality Checkpoints
**Stop and validate** at these moments:
- After implementing a complete feature
- Before starting a new major component  
- When something feels wrong
- Before declaring "done"
- **WHEN HOOKS FAIL WITH ERRORS** ❌

Run appropriate validation commands for the project (e.g., linting, testing)

> Why: You can lose track of what's actually working. These checkpoints prevent cascading failures.

### 🚨 CRITICAL: Hook Failures Are BLOCKING
**When hooks report ANY issues (exit code 2), you MUST:**
1. **STOP IMMEDIATELY** - Do not continue with other tasks
2. **FIX ALL ISSUES** - Address every ❌ issue until everything is ✅ GREEN
3. **VERIFY THE FIX** - Re-run the failed command to confirm it's fixed
4. **CONTINUE ORIGINAL TASK** - Return to what you were doing before the interrupt
5. **NEVER IGNORE** - There are NO warnings, only requirements

This includes:
- Formatting issues (gofmt, black, prettier, etc.)
- Linting violations (golangci-lint, eslint, etc.)
- Forbidden patterns (time.Sleep, panic(), interface{})
- ALL other checks

Your code must be 100% clean. No exceptions.

**Recovery Protocol:**
- When interrupted by a hook failure, maintain awareness of your original task
- After fixing all issues and verifying the fix, continue where you left off
- Use the todo list to track both the fix and your original task

## Working Memory Management

### When context gets long:
- Re-read this CLAUDE.md file
- Summarize progress in a PROGRESS.md file
- Document current state before major changes

### Maintain TODO.md:
```
## Current Task
- [ ] What we're doing RIGHT NOW

## Completed  
- [x] What's actually done and tested

## Next Steps
- [ ] What comes next
```

## Python-Specific Rules

### FORBIDDEN - NEVER DO THESE:
- **NO** generic `except:` clauses - always specify exception types!
- **NO** mutable default arguments in functions
- **NO** keeping old and new code together
- **NO** migration functions or compatibility layers
- **NO** versioned function names (process_v2, handle_new)
- **NO** complex inheritance hierarchies
- **NO** TODOs in final code

> **AUTOMATED ENFORCEMENT**: Linters and hooks will BLOCK commits that violate these rules.  
> When you see `❌ FORBIDDEN PATTERN`, you MUST fix it immediately!

### Required Standards:
- **Delete** old code when replacing it
- **Meaningful names**: `user_id` not `id`
- **Early returns** to reduce nesting
- **Type hints** for all functions: `def process(data: dict) -> str:`
- **Docstrings** for all public functions and classes
- **Simple exceptions**: `raise ValueError(f"Invalid input: {value}")`
- **pytest** for all tests
- **Path objects** for file operations: Use `pathlib.Path` not string concatenation

## Implementation Standards

### Our code is complete when:
- ✓ All linters pass with zero issues
- ✓ All tests pass  
- ✓ Feature works end-to-end
- ✓ Old code is deleted
- ✓ Docstrings on all public functions/classes

### Testing Strategy
- Complex business logic → Write tests first
- Simple utilities → Write tests after
- Performance critical → Add benchmarks
- Skip tests for simple CLI argument parsing

### Project Structure
```
synctalk/       # Main package with modular components
scripts/        # Standalone scripts (training, inference)
data_utils/     # Data preprocessing utilities
tests/          # Test suite
docs/           # Documentation
demo/           # Demo files and examples
checkpoint/     # Model checkpoints
dataset/        # Training datasets
```

## Problem-Solving Together

When you're stuck or confused:
1. **Stop** - Don't spiral into complex solutions
2. **Delegate** - Consider spawning agents for parallel investigation
3. **Ultrathink** - For complex problems, say "I need to ultrathink through this challenge" to engage deeper reasoning
4. **Step back** - Re-read the requirements
5. **Simplify** - The simple solution is usually correct
6. **Ask** - "I see two approaches: [A] vs [B]. Which do you prefer?"

My insights on better approaches are valued - please ask for them!

## Performance & Security

### **Measure First**:
- No premature optimization
- Benchmark before claiming something is faster
- Use cProfile or line_profiler for real bottlenecks
- Memory profiling with memory_profiler when needed

### **Security Always**:
- Validate all inputs (especially file paths)
- Use secrets module for secure randomness
- Never use eval() or exec() with user input
- Sanitize paths to prevent directory traversal

## Communication Protocol

### Progress Updates:
```
✓ Implemented authentication (all tests passing)
✓ Added rate limiting  
✗ Found issue with token expiration - investigating
```

### Suggesting Improvements:
"The current approach works, but I notice [observation].
Would you like me to [specific improvement]?"

## Working Together

- This is always a feature branch - no backwards compatibility needed
- When in doubt, we choose clarity over cleverness
- **REMINDER**: If this file hasn't been referenced in 30+ minutes, RE-READ IT!

Avoid complex abstractions or "clever" code. The simple, obvious solution is probably better, and my guidance helps you stay focused on what matters.

## Project Understanding Protocol

### Universal Documentation System
When working on any codebase, maintain a **PROJECT_DOCS.md** file to preserve context across sessions. This prevents repeated analysis and ensures efficient navigation.

### Documentation Structure
Create PROJECT_DOCS.md with these sections:

#### 1. ARCHITECTURE OVERVIEW
```markdown
## System Architecture
- **Purpose**: [What does this system do?]
- **Core Approach**: [How does it work at highest level?]
- **Key Technologies**: [Main tech stack]

## Data Flow
[Input] → [Processing] → [Output]

## Major Components
- Component A: [Purpose and location]
- Component B: [Purpose and location]
```

#### 2. COMPONENT MAP
```markdown
## Component Dependencies
A → B → C (explain relationship)

## Entry Points
- Main: [file:line] - [purpose]
- CLI: [file:line] - [purpose]
- API: [file:line] - [purpose]

## Configuration System
- Config files: [location and purpose]
- Environment vars: [key ones]
```

#### 3. KEY PATTERNS
```markdown
## Code Patterns
- Pattern 1: [description] - Example: [file:line]
- Pattern 2: [description] - Example: [file:line]

## Common Abstractions
- [Abstraction]: Used for [purpose]

## Anti-patterns to Avoid
- Don't do X, instead do Y
```

#### 4. NAVIGATION GUIDE
```markdown
## Quick Find
- Feature X implementation: [file:line]
- Configuration for Y: [file:line]
- Tests for Z: [file:line]

## Directory Purposes
- /src: [what goes here]
- /lib: [what goes here]
```

#### 5. TASK PLAYBOOK
```markdown
## Common Tasks

### Adding a new feature
1. Check [file] for patterns
2. Modify [component]
3. Update [config]

### Debugging issues
1. Start at [entry point]
2. Check [logs location]
3. Common issues: [list]
```

### Maintenance Protocol

**When to Update PROJECT_DOCS.md:**
- After discovering major architectural patterns
- When completing significant features
- After resolving complex debugging sessions
- When you find yourself re-analyzing the same code

**Update Process:**
1. Add new discoveries to relevant sections
2. Update file:line references if code moves
3. Document any gotchas or non-obvious behaviors
4. Keep it concise - reference code, don't duplicate it

### Usage Workflow

1. **Starting work**: Check for PROJECT_DOCS.md first
2. **If missing**: Create it during initial exploration
3. **During work**: Reference it instead of re-analyzing
4. **After changes**: Update affected sections
5. **Before context switch**: Ensure recent discoveries are documented

This system ensures efficient context preservation and reduces repeated analysis across sessions.
