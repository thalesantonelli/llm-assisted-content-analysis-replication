# Environmental and Climate Repertoire

## Task

Determine whether the parliamentary speech mobilizes an environmental, climate, nature, or sustainability-related repertoire.

## Definition

Classify as **PRESENCE** whenever the speech establishes an explicit or identifiable implicit connection with environmental or climate issues.

The classification concerns the presence of this repertoire, regardless of whether the speaker supports, opposes, criticizes, instrumentalizes, or strategically invokes it.

Relevant references include:

- climate change, global warming, climate crisis, or environmental disasters;
- environmental protection or conservation;
- forests, biomes, rivers, fauna, or flora;
- deforestation, fires, pollution, environmental degradation, or natural-resource exploitation;
- sustainable development, environmental sustainability, green economy, or ecological transition;
- environmental agreements, commitments, policies, or international agendas;
- environmental impacts associated with economic activities, public policies, or development models;
- strategic or instrumental uses of environmental arguments;
- technocratic, economic, or business-oriented environmental claims;
- greenwashing or rhetorical environmentalism.

## Absence

Classify as **ABSENCE** when:

1. no identifiable environmental, climate, nature, or sustainability-related element is present;
2. references to development, progress, or the future have no discernible environmental meaning;
3. the speech concerns land, political, or institutional conflicts without mobilizing an environmental or climate-related argument.

## Decision rule

Does the speech directly or indirectly mobilize an identifiable environmental, climate, nature, or sustainability-related element?

If yes, classify as **PRESENCE**, regardless of the speaker's normative position.

Otherwise, classify as **ABSENCE**.

## Input

{text}

## Output

Return only a valid JSON object:

{
  "environmental_repertoire": "presence" or "absence",
  "justification": "brief and objective justification"
}
