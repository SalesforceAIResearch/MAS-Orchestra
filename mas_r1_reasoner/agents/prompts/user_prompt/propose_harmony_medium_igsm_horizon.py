

base = """
Please solve the given question by creating one or more agents and connecting them into a valid computational graph that collaboratively produces the final answer. To create an agent, you must define that agent by outputting <agent> with agent_id (a unique id for the agent, must be unique and contain only alphanumeric or underscore characters (e.g., A1, Refine_1, WS_Japan)), agent_name (exactly one of: CoTAgent), agent_description, required_arguments (must include at least one <agent_input> tag. DebateAgents must define <debate_roles> with two or more roles. If  <agent_input> left empty (""), the parser will automatically replace it with the original question.). 

After defining all agents, you must build a valid graph by specifying edges that describe the data flow between agents. Output exactly one <edge> block, Each <from> <to> pair connects the output of one agent to the input of another: 

<edge>
<from>source_agent_id</from>
<to>target_agent_id</to>
</edge>

You can output multiple <from> and </to> inside the <edge>. Each <from> and <to> value must exactly match an existing <agent_id>.

To be valid, your graph must satisfy all of the following constraints:

1. Node consistency: Every <from> and <to> must reference a valid <agent_id> that appears in an <agent> block.
2. Directionality: Edges are directed: data flows from <from> → <to>.
3. Connectivity: Every agent must be connected directly or indirectly to the main flow. Isolated agents are not allowed.
4. Start node(s): At least one agent must have no incoming edge. These are “entry points” (e.g., initial reasoning).
5. Sink node: There must be exactly one agent with no outgoing edge — this is the FINAL agent that produces the answer.
6. No undefined edges: It is invalid to reference an agent in <from> or <to> that was not declared.
7. No loops or cycles: No self-loop: <from>X</from><to>X</to> is not allowed. No cycles: The graph must be acyclic; a topological order must exist.
8. Parallelism allowed: Multiple agents may have the same <from> or <to> (fan-out/fan-in).
9. Unambiguous sink: The parser will reject graphs with multiple sinks (add a final “collector” agent if needed).
10. Order-independent: The XML order of edges does not need to follow execution order; topological sorting is handled automatically.
11. Sink answer completeness: The unique sink agent's output must directly answer the original question in a user-ready form. It must not be an intermediate artifact (e.g., notes, critique, raw table) unless the question explicitly asks for that artifact. If <agent_input> is empty for the sink, it inherits the original question and must return the final answer. If <agent_input> is non-empty, the runner still prepends the original question as context; the sink must still produce the final, user-facing answer to that question.
12. Edge-Data Flow Consistency (BIDIRECTIONAL): Edges represent execution order. ${} represents data flow. As a result, you must ensure they are consistent with each other.
    a) If an <agent_input> references ${X}, there MUST be an edge <from>X</from><to>THIS_AGENT</to>
    b) If there is an edge <from>X</from><to>Y</to>, then Y's <agent_input> MUST reference ${X}
    In other words: edges exist if and only if there is actual data passing from one agent to another. Do not create edges solely for execution ordering without data flow.

Thinking Section (Required):

Before defining agents and edges, you must include a <thinking> section.
It should naturally describe why multiple agents are needed, why each type was chosen, and why the graph has that structure (parallel, sequential or hybrid).
It must justify both planning and design rationale.

Example structure:

<thinking>
  Explain why a single agent is insufficient.
  Describe each agent's role and how they connect.
  Justify the flow pattern (parallel, sequential, hybrid).
  End by clearly stating which agent produces the final output.
</thinking>


Single-agent example:

If you decide to solve via single agent, you will output the following. In this case, since the agent_input is the same as the original task, you must set the agent_input as empty (""), and the parser will replace it with the original question.

1st example:

Question: The number of each Patella's Melanocytes equals 20 more than the difference of each Patella's Osteoblasts and each Euglena's Patella. The number of each Euglena's Patella equals 16. The number of each Patella's Osteoblasts equals 0. The number of each Rotifer's Biceps equals 3 more than each Patella's Osteoblasts. The number of each Biceps's Osteoblasts equals 13 times as much as each Patella's Osteoblasts. How many Organs does Euglena have?

  <thinking>
    This is a small algebraic system. One Chain-of-Thought (CoT) pass can parse and solve it directly and report the final value.
    The final answer to the original question will be the output of the CoTAgent.
  </thinking>
  <agent>
    <agent_id>Reason_Euglena_Organs</agent_id>
    <agent_name>CoTAgent</agent_name>
    <agent_description>Parse and solve the system in one pass; return the requested count.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
      </required_arguments>
  </agent>

Multi-Agent Example: 

If you decide to solve via multiple agents, you must first decompose the original question into smaller, well-defined sub-tasks, each representing a single sub-goal. Then, create one agent per sub-task and connect them into a coherent, acyclic computational graph.

In this case, the agent_input is not empty and serves as the specific sub-task for the agent to solve, while the parser automatically prepends the original question as context before the provided agent_input content.

When decomposing:

1. Keep sub-tasks minimal and focused. Each agent should handle one atomic objective (e.g., one query, one reasoning step, or one verification task).
2. Connect agents logically so that information flows toward a single final agent (the sink) that directly answers the original question.
3. If the question provides explicit sub-tasks (for example, problem 1 and 2... are provided), you should create multiple agents accordingly and solve each sub-task with its own agent.


1st example:

Question: The number of each Duffle Backpack's Watercolor Pencil equals each Computer Science Lab's Backpack. The number of each Duffle Backpack's Brush Pen equals 4 times as much as each Compression Backpack's Watercolor Pencil. The number of each Compression Backpack's Brush Pen equals 2 times as much as each Compression Backpack's Watercolor Pencil. The number of each Molecular Biology Lab's Duffle Backpack equals 11 more than each Computer Science Lab's Compression Backpack. The number of each Computer Science Lab's Duffle Backpack equals 1 times as much as the difference of each Compression Backpack's Stationery and each Compression Backpack's Brush Pen. The number of each Compression Backpack's Watercolor Pencil equals 13. The number of each Computer Science Lab's Compression Backpack equals 2 more than each Compression Backpack's Stationery. How many Stationery does Molecular Biology Lab have?

<thinking>
  A single agent could attempt both information extraction and solving, but this mix often causes subtle errors (missed variables, sign flips, or unit mix-ups). 
  We split the task into four narrow roles:
  (1) Parse_Backpack_Equations converts the story into a clean equation set so later steps do not rely on ad-hoc re-parsing. 
  (2) Solve_MB_Stationery applies self-consistency to stabilize algebra on a small but coupled system; multiple samples reduce the chance of a one-off substitution error. 
  (3) QA_MB_Stationery_Verify checks every original statement and integer constraints; this separates verification from computation to avoid confirmation bias. 
  (4) Final_MB_Stationery writes the user-ready answer using only verified values.
  The flow is mostly sequential with a small fan-out into QA for independent checking. Parse_* is the start node; Final_* is the unique sink.
</thinking>

<agent>
  <agent_id>Parse_Backpack_Equations</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Define concise symbols (e.g., CP_ST, CP_WP, CP_BP, CS_DB, CS_CB, DF_WP, DF_BP, MB_DB, etc.) and list all equations in simplified form with a note on nonnegative integers.</agent_description>
  <required_arguments>
    <agent_input>Extract variables and return one equation per line, simplified.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_MB_Stationery</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve for CP_ST first, then derive MB_DB (or any needed bridge) to answer “How many Stationery does Molecular Biology Lab have?”. Use self-consistency reasoning with multiple checks.</agent_description>
  <required_arguments>
    <agent_input>Use the equations below to compute the target value step by step. Return the derived tuple of required variables and one-line rationale.

Equations:
${Parse_Backpack_Equations}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>QA_MB_Stationery_Verify</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Verify that the solved solution satisfies every equation and the integer nonnegativity constraint. Propose minimal fixes if needed.</agent_description>
  <required_arguments>
    <agent_input>Check each original equation with the solved values; confirm nonnegative integers; confirm target computability. Return “OK” or a corrected tuple with a brief note.

Equations:
${Parse_Backpack_Equations}

Solved values:
${Solve_MB_Stationery}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Final_MB_Stationery</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Produce the final numeric answer for Molecular Biology Lab's Stationery with one key supporting substitution.</agent_description>
  <required_arguments>
    <agent_input>State the final value clearly and include one short supporting step.

Solution:
${Solve_MB_Stationery}

QA:
${QA_MB_Stationery_Verify}</agent_input>
  </required_arguments>
</agent>

<edge>
  <from>Parse_Backpack_Equations</from><to>Solve_MB_Stationery</to>
  <from>Parse_Backpack_Equations</from><to>QA_MB_Stationery_Verify</to>
  <from>Solve_MB_Stationery</from><to>QA_MB_Stationery_Verify</to>
  <from>Solve_MB_Stationery</from><to>Final_MB_Stationery</to>
  <from>QA_MB_Stationery_Verify</from><to>Final_MB_Stationery</to>
</edge>


2nd example:
Question: The number of each Red Eyed Tree Frog's Primary Feathers equals 2 times as much as each Kelp Forest Tank's Red Eyed Tree Frog. The number of each Ngorongoro Conservation Area's Kelp Forest Tank equals each Kelp Forest Tank's Red Eyed Tree Frog. The number of each Kruger National Park's Kelp Forest Tank equals 12. The number of each Toad's Coracoid equals each Kelp Forest Tank's Red Eyed Tree Frog. The number of each Ngorongoro Conservation Area's Rockpool Exhibit equals 21 more than the difference of each Kruger National Park's Kelp Forest Tank and each Rockpool Exhibit's Red Eyed Tree Frog. The number of each Rockpool Exhibit's Red Eyed Tree Frog equals 17 more than each Kruger National Park's Kelp Forest Tank. The number of each Toad's Primary Feathers equals 10. The number of each Rockpool Exhibit's Toad equals the sum of each Toad's Primary Feathers, each Ngorongoro Conservation Area's Kelp Forest Tank and each Kelp Forest Tank's Red Eyed Tree Frog. The number of each Kelp Forest Tank's Red Eyed Tree Frog equals 18 times as much as each Toad's Primary Feathers. The number of each Kelp Forest Tank's Toad equals each Kruger National Park's Rockpool Exhibit. The number of each Red Eyed Tree Frog's Coracoid equals the sum of each Ngorongoro Conservation Area's Animal, each Toad's Bone and each Toad's Primary Feathers. The number of each Kruger National Park's Rockpool Exhibit equals 22 more than each Rockpool Exhibit's Toad. How many Kelp Forest Tank does Kruger National Park have?

<thinking>
  The story contains many cross-references; inline solving risks mixing roles (extraction vs. computation) and losing track of dependencies. 
  We therefore assign:
  (1) Parse_KFT_Equation_Set to create a single, authoritative equation set and variable map. 
  (2) Solve_KNP_KFT to compute the target deterministically. 
  (3) QA_KNP_KFT_Verify to check every original statement and integer constraints independently of the solver. 
  (4) Final_KNP_KFT to produce the final user-facing value with a short supporting step.
  This separation reduces error propagation and makes failure modes easier to diagnose. Parse_* is the entry; Final_* is the unique sink.
</thinking>

<agent>
  <agent_id>Parse_KFT_Equation_Set</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Define symbols (e.g., KFT_REF, NCA_KFT, KNP_KFT, RPE_REF, RPE_Toad, etc.) and list all equations in simplified form with nonnegative integers noted.</agent_description>
  <required_arguments>
    <agent_input>Return compact variable definitions and one equation per line.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_KNP_KFT</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Compute KNP_KFT from the parsed equations with explicit intermediate derivations.</agent_description>
  <required_arguments>
    <agent_input>Use the parsed equations to solve for KNP_KFT. Return the derived value and any required intermediate variables with a one-line rationale.

Equations:
${Parse_KFT_Equation_Set}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>QA_KNP_KFT_Verify</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Verify that the solved solution satisfies all equations and the integer domain; suggest minimal corrections if needed.</agent_description>
  <required_arguments>
    <agent_input>Check each original statement against the solved result. Return “OK” or a corrected tuple with a brief note.

Equations:
${Parse_KFT_Equation_Set}

Solved result:
${Solve_KNP_KFT}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Final_KNP_KFT</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Return Kruger National Park's Kelp Forest Tank count with one supporting step.</agent_description>
  <required_arguments>
    <agent_input>Answer in one sentence and include one short supporting equality.

Solution:
${Solve_KNP_KFT}

QA:
${QA_KNP_KFT_Verify}</agent_input>
  </required_arguments>
</agent>

<edge>
  <from>Parse_KFT_Equation_Set</from><to>Solve_KNP_KFT</to>
  <from>Parse_KFT_Equation_Set</from><to>QA_KNP_KFT_Verify</to>
  <from>Solve_KNP_KFT</from><to>QA_KNP_KFT_Verify</to>
  <from>Solve_KNP_KFT</from><to>Final_KNP_KFT</to>
  <from>QA_KNP_KFT_Verify</from><to>Final_KNP_KFT</to>
</edge>


3rd example:

Question: The number of each Governor's District's Family Dollar equals the sum of each Dollar General's Rice Cakes, each Aldi's Crackers, each Governor's District's Aldi and each Municipal District's Family Dollar. The number of each Capitol District's Aldi equals 22. The number of each Municipal District's Family Dollar equals 2 times as much as each Governor's District's Dollar General. The number of each Capitol District's Family Dollar equals each Municipal District's Family Dollar. The number of each Municipal District's Dollar General equals 20 times as much as each Governor's District's Family Dollar. The number of each Municipal District's Aldi equals each Family Dollar's Crackers. The number of each Aldi's Chocolate Covered Pretzels equals 2. The number of each Dollar General's Chocolate Covered Pretzels equals the sum of each Governor's District's Aldi, each Aldi's Rice Cakes, each Dollar General's Rice Cakes and each Governor's District's Dollar General. The number of each Aldi's Crackers equals 18 more than each Governor's District's Aldi. The number of each Family Dollar's Chocolate Covered Pretzels equals 9 times as much as the sum of each Aldi's Chocolate Covered Pretzels and each Capitol District's Aldi. The number of each Family Dollar's Crackers equals 22. The number of each Aldi's Rice Cakes equals the sum of each Dollar General's Rice Cakes, each Governor's District's Aldi, each Governor's District's Dollar General and each Aldi's Crackers. The number of each Governor's District's Aldi equals 0. The number of each Governor's District's Dollar General equals 18 times as much as each Aldi's Crackers. The number of each Dollar General's Crackers equals 1. The number of each Dollar General's Rice Cakes equals 8 more than the sum of each Aldi's Crackers and each Governor's District's Aldi. How many Rice Cakes does Dollar General have?

<thinking>
  This system is dense and easy to mis-handle if extraction, solving, and validation are mixed. 
  We design five roles:
  (1) Parse_DG_System produces a stable equation set with clear variable names so all later steps share the same ground truth. 
  (2) Solve_DG_RC_Algebra uses elimination to derive the target; this path favors closed-form steps. 
  (3) Solve_DG_RC_Substitution uses a direct substitution chain from constants; this path favors constructive derivation. 
  (4) Debate_Select_DG_RC compares both candidates and picks the one that matches more equations and domain constraints; this reduces single-path bias. 
  (5) QA_DG_RC_Verify checks the chosen value against all original statements before the final write-up. 
  The graph fans out after parsing, converges in debate, and then passes through verification to a single sink (Final_DG_RiceCakes). This hybrid pattern balances accuracy and traceability.
</thinking>

<agent>
  <agent_id>Parse_DG_System</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Define canonical variables (e.g., GD_FD, GD_ALDI, MD_FD, MD_DG, DG_RC, ALD_CR, etc.) and list simplified equations, noting nonnegative integers.</agent_description>
  <required_arguments>
    <agent_input>Return the variable list and one equation per line, simplified.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_DG_RC_Algebra</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Derive DG_RC using algebraic elimination with brief key steps.</agent_description>
  <required_arguments>
    <agent_input>Using the parsed system, eliminate to a direct expression for DG_RC. Return the candidate value and minimal supporting steps.

Equations:
${Parse_DG_System}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_DG_RC_Substitution</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Reach DG_RC by iterative substitution from constants and direct equalities.</agent_description>
  <required_arguments>
    <agent_input>Starting from the simplest links, substitute to compute DG_RC. Return the candidate value and minimal steps.

Equations:
${Parse_DG_System}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Debate_Select_DG_RC</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Compare the two candidate values and pick one that is fully consistent with the system.</agent_description>
  <required_arguments>
    <agent_input>Compare which candidate respects all equations and the integer domain. Choose a single value and justify briefly.

Algebra path:
${Solve_DG_RC_Algebra}

Substitution path:
${Solve_DG_RC_Substitution}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>QA_DG_RC_Verify</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Validate the debated result against every original equation; propose a minimal correction if needed.</agent_description>
  <required_arguments>
    <agent_input>Verify the chosen DG_RC against all equations and nonnegative integer constraints. Return “OK” or a corrected value with a short note.

Parsed equations:
${Parse_DG_System}

Debated result:
${Debate_Select_DG_RC}</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Final_DG_RiceCakes</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>State the final value for Dollar General's Rice Cakes with one supporting equality.</agent_description>
  <required_arguments>
    <agent_input>Answer: “Dollar General has [DG_RC] Rice Cakes,” and include one short supporting step.

Debate:
${Debate_Select_DG_RC}

QA:
${QA_DG_RC_Verify}</agent_input>
  </required_arguments>
</agent>

<edge>
  <from>Parse_DG_System</from><to>Solve_DG_RC_Algebra</to>
  <from>Parse_DG_System</from><to>Solve_DG_RC_Substitution</to>
  <from>Solve_DG_RC_Algebra</from><to>Debate_Select_DG_RC</to>
  <from>Solve_DG_RC_Substitution</from><to>Debate_Select_DG_RC</to>
  <from>Parse_DG_System</from><to>QA_DG_RC_Verify</to>
  <from>Debate_Select_DG_RC</from><to>QA_DG_RC_Verify</to>
  <from>Debate_Select_DG_RC</from><to>Final_DG_RiceCakes</to>
  <from>QA_DG_RC_Verify</from><to>Final_DG_RiceCakes</to>
</edge>

4th example:

Question: 
Problem 1: Let X=10. What is X?
Problem 2: Let Y=[answer1]+3. What is Y?
Problem 3: Let Z=2x[answer2]-[answer1]. What is Z?

Output as 
### Final Answers

Problem 1: \boxed{[answer1]}
Problem 2: \boxed{[answer2]}
Problem 3: \boxed{[answer3]}

<thinking>
  The task has three explicit sub-problems with dependencies. 
  Each solver should see only what it needs:
  - Solve_P1_X reads Problem 1 only.
  - Solve_P2_Y reads Problem 2 and the numeric value of [answer1] from Problem 1.
  - Solve_P3_Z reads Problem 3 and the numeric values of [answer1] and [answer2].
  A QA agent checks that dependencies were respected. 
  The sink Final_Simple_Boxed receives all sub-answers and formats the output. It does not recompute.
</thinking>

<agent>
  <agent_id>Solve_P1_X</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 1 with only Problem 1 context.</agent_description>
  <required_arguments>
    <agent_input>Problem 1: Let X = 10. What is X?
Return: answer1 = 10.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P2_Y</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 2 using only Problem 2 text and [answer1].</agent_description>
  <required_arguments>
    <agent_input>Problem 2: Let Y = [answer1] + 3. What is Y?
Given: [answer1] = ${Solve_P1_X}
Compute: answer2 = [answer1] + 3 = 10 + 3 = 13.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Solve_P3_Z</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Solve Problem 3 using only Problem 3 text plus [answer1], [answer2].</agent_description>
  <required_arguments>
    <agent_input>Problem 3: Let Z = 2 × [answer2] − [answer1]. What is Z?
Given: [answer1] = ${Solve_P1_X}, [answer2] = ${Solve_P2_Y}
Compute: answer3 = 2 × 13 − 10 = 16.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>QA_Simple_Dependency_Check</agent_id>
  <agent_name>ReflexionAgent</agent_name>
  <agent_description>Verify P2 used [answer1]=10 and P3 used [answer1]=10, [answer2]=13; arithmetic is correct.</agent_description>
  <required_arguments>
    <agent_input>Answers:
answer1 = ${Solve_P1_X}
answer2 = ${Solve_P2_Y}
answer3 = ${Solve_P3_Z}
Check dependencies and arithmetic; return "OK" if consistent.</agent_input>
  </required_arguments>
</agent>

<agent>
  <agent_id>Final_Simple_Boxed</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>Render the three boxed answers using only upstream results.</agent_description>
  <required_arguments>
    <agent_input>Use these values:
answer1 = ${Solve_P1_X}
answer2 = ${Solve_P2_Y}
answer3 = ${Solve_P3_Z}

Produce EXACTLY:

### Final Answers

Problem 1: \\boxed{${Solve_P1_X}}
Problem 2: \\boxed{${Solve_P2_Y}}
Problem 3: \\boxed{${Solve_P3_Z}}</agent_input>
  </required_arguments>
</agent>

<edge>
  <from>Solve_P1_X</from><to>Solve_P2_Y</to>
  <from>Solve_P1_X</from><to>Solve_P3_Z</to>
  <from>Solve_P2_Y</from><to>Solve_P3_Z</to>

  <from>Solve_P1_X</from><to>QA_Simple_Dependency_Check</to>
  <from>Solve_P2_Y</from><to>QA_Simple_Dependency_Check</to>
  <from>Solve_P3_Z</from><to>QA_Simple_Dependency_Check</to>

  <from>Solve_P1_X</from><to>Final_Simple_Boxed</to>
  <from>Solve_P2_Y</from><to>Final_Simple_Boxed</to>
  <from>Solve_P3_Z</from><to>Final_Simple_Boxed</to>
  <from>QA_Simple_Dependency_Check</from><to>Final_Simple_Boxed</to>
</edge>


Final Checklist:
1. Unique IDs: Every <agent_id> is unique and uses only letters, digits, or underscores (A1, WS_Grid, Final_1, etc.). Bellow is WRONG as there are two FINAL agents:
<agent>
  <agent_id>FINAL</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<agent>
  <agent_id>FINAL</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>
2. Declared First	All <agent> blocks appear before any <edge> definitions.
3. Connected Graph	No isolated agents — each must connect directly or indirectly to the main flow. Bellow is WRONG as B is isolated agent: 
<agent>
  <agent_id>A</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<agent>
  <agent_id>B</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<edge>
  <from>A</from><to>FINAL</to>
</edge>
4. At Least One Start Node:	At least one agent has no incoming edge (e.g., initial reasoning).
5. Exactly One Sink Node	Exactly one agent has no outgoing edge — this is the final output (sink). Bellow is WRONG as there are two sink nodes (B and C):
<edge>
  <from>A</from><to>B</to>
  <from>A</from><to>C</to>
</edge>
6. No Dangling Edges:	Every <from> and <to> exactly matches an existing <agent_id>. Bellow is WRONG as B is not an existing <agent_id>: 
<agent>
  <agent_id>A</agent_id>
  <agent_name>CoTAgent</agent_name>
  <agent_description>...</agent_description>
  <required_arguments><agent_input></agent_input></required_arguments>
</agent>

<edge>
  <from>A</from><to>B</to>
</edge>
7. No Duplicates in Edges: Do not define the same <from> <to> link twice. Bellow is WRONG as it defines the same <from> <to> link twice:
<edge>
  <from>A</from><to>B</to>
  <from>A</from><to>B</to>
</edge>
8. Parallel Allowed	Fan-out and fan-in are fine (e.g., multiple <from>s into one <to>).
9. Edge-Data Flow Consistency: Every edge must correspond to actual data flow. If <from>X</from><to>Y</to> exists, then Y's <agent_input> MUST include ${X}. Conversely, if Y uses ${X}, you MUST have an edge <from>X</from><to>Y</to>. Below is WRONG because edge C→D exists but D doesn't use ${C}:
<agent>
  <agent_id>C</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input>Analyze data.</agent_input></required_arguments>
</agent>
<agent>
  <agent_id>D</agent_id>
  <agent_name>CoTAgent</agent_name>
  <required_arguments><agent_input>Make final decision.</agent_input></required_arguments>
</agent>
<edge>
  <from>C</from><to>D</to>
</edge>
The correct version should have D use ${C}: <agent_input>Make final decision based on: ${C}</agent_input>
10. Consistent Agent Types:	Each agent uses only one of the allowed names: CoTAgent.
11. No Self-Loops: Never link an agent to itself. No Cycles: The graph must be a DAG (topological order exists). Bellow is WRONG as there is a cycle (A -> B -> C -> A) and cycle (REFLECT -> REFLECT):
<edge>
  <from>A</from><to>B</to>
  <from>B</from><to>C</to>
  <from>C</from><to>A</to>
  <from>REFLECT</from><to>REFLECT</to>
</edge>


Below is the question to solve:\n\n[QUESTION]
"""

