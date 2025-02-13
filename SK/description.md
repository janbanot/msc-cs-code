# Political Media Influence Simulation

## Core Concept
This NetLogo model simulates how media outlets and social networks influence political opinions. It demonstrates:
- Propaganda mechanisms (left/right-leaning media)
- Social contagion of political views
- Fatigue effects from constant persuasion attempts
- Economic/event-driven external influences

## Key Components

### Agents
- **Users**: Citizens with:
  - Political views (left/right/undecided)
  - Susceptibility to influence
  - Opinion change history
  - Fatigue accumulation
- **Media**: Outlets with:
  - Political alignment (left/right/neutral)
  - Reach capacity
  - Persuasion effectiveness
  - Success tracking

### Network Structure
- Small-world social network (Watts-Strogatz)
- Media-to-user connections based on reach
- Dual influence channels:
  - Direct media propaganda
  - Peer-to-peer social influence

### Dynamic Systems
- Opinion change thresholds
- Fatigue accumulation (resistance build-up)
- Random economic fluctuations
- Periodic impactful events

## Core Mechanisms

### Media Influence
- Propaganda outlets push specific agendas
- Effectiveness combines:
  - Media's inherent strength
  - Target's current fatigue
  - Network proximity
- Neutral media provide balanced information

### Social Influence
- Users observe neighbors' opinions
- Majority views create peer pressure
- Threshold-based conversion (55-90% neighbor consensus)

### Fatigue System
- Cumulative resistance to persuasion
- Reduces media effectiveness over time
- Configurable accumulation rate

### External Factors
- Economic conditions modifier
- Random events (10% chance/tick)
- Three event types with variable impact

## Metrics Tracked
- Political distribution (% left/right/undecided)
- Polarization index (0=balanced, 1=extremes)
- Opinion change statistics
- Media performance metrics:
  - Most influential outlet
  - Comparative effectiveness (propaganda vs neutral)

## Implementation Details
1. **Network Generation**
   - Uses NW extension for small-world topology
   - Media connections scale with reach parameter
   - Circular layout for visualization clarity

2. **Opinion Calculus**
   - Media effects accumulate additively
   - Fatigue creates diminishing returns
   - Threshold comparison uses absolute values

3. **Statistical Tracking**
   - Maintains both aggregate and per-agent histories
   - Calculates polarization using decided voters only
   - Normalizes media effectiveness comparisons

## Potential Use Cases
- Studying filter bubble formation
- Testing media regulation strategies
- Analyzing polarization drivers
- Comparing propaganda vs organic spread
- Modeling resistance to misinformation
