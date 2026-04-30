"""
From Chat: Causal inference steps (Propensity Score Matching for pre/post):

Define treatment: teams that moved to the new tiered pricing (P=1) vs. teams that stayed on the old flat price (P=0)
Select covariates (X): company size, industry, current seat count, historical token usage, account age, region
Estimate propensity scores: fit a logistic regression where P(treatment=1 | X) gives each team a propensity score
Match: pair each treated team with the closest untreated team(s) by propensity score (nearest neighbor, caliper matching, or inverse propensity weighting)
Validate balance: confirm that post-matching, the distribution of X is similar across treatment and control (standardized mean differences < 0.1)
Estimate treatment effect: compare Y (gross profit, seats, token usage) between matched treated and control groups using a difference-in-differences framework (pre vs. post, treated vs. matched control), which nets out time trends

The key insight for the interview: propensity matching alone only controls for observables. Diff-in-diff on top of it helps with unobserved time-invariant confounders. Flag this limitation proactively.

From Code:
The three notebooks most relevant to your question are:                             
                                                                                                                                                                             
  - 05-Propensity-Score.ipynb — PSM mechanics, IPW, doubly robust estimation                                                                                                 
  - 08-Difference-in-Differences.ipynb — pre-post analysis, and crucially, DID combined with propensity scores at the end                                                    
  - 09-Synthetic-Control.ipynb — what you're saying you can't use                                                                                                            
                                                                                                                                                                             
  ---                                                                                                                                                                        
  PSM + Pre-Post in simple steps                                                                                                                                             
                                                                                                                                                                             
  Here's the scenario: something happened (a treatment), you have before/after data, and you have units that got treated and units that didn't — but the groups aren't       
  comparable. You can't build a synthetic control (maybe too few control units, short pre-period, or too much heterogeneity).                                                
                  
  The intuition in one sentence: Match treated units to similar untreated units first, then compare how each group changed over time.                                        
                  
  The 5 steps                                                                                                                                                                
                  
  1. Collect pre-treatment covariates — characteristics measured before the intervention that predict both who gets treated and the outcome. (Same confounders you'd put in a
   DAG — notebook 03 covers this.)
  2. Estimate propensity scores — fit a logistic regression: P(treated | covariates). Each unit gets a single number summarizing "how likely were they to be treated?" This  
  collapses a high-dimensional covariate space into one dimension. (Notebook 05, first section.)                                                                             
  3. Match — for each treated unit, find the untreated unit(s) with the closest propensity score. Now you have paired groups that look similar on observables. (Notebook 05
  uses KNN matching.)                                                                                                                                                        
  4. Check balance — verify the matched groups actually have similar covariate distributions. If they don't, your matching failed and the estimate will be biased. (Notebook
  05 visualizes this.)                                                                                                                                                       
  5. Compute the DID on matched pairs — for each matched pair, calculate (treated_post - treated_pre) - (control_post - control_pre). Average that across pairs. That's your
  treatment effect estimate.                                                                                                                                                 
                  
  Why this works                                                                                                                                                             
                  
  Standard pre-post (just comparing before/after in the treated group) is biased by any time trend — maybe everyone improved, not just the treated. Plain DID fixes that by  
  subtracting the control group's trend, but assumes parallel trends: absent treatment, both groups would have moved the same way.
                                                                                                                                                                             
  That parallel trends assumption is much more believable when treated and control units are similar. PSM's job is to make them similar so that the parallel trends          
  assumption holds.
                                                                                                                                                                             
  ★ Insight ─────────────────────────────────────                                                                                                                            
  Think of it as two layers of bias removal. 
  - PSM handles selection bias (treated units differ from control units). 
  - DID handles time bias (things changed for everyone).
  Neither alone is sufficient.
  PSM without DID can't distinguish treatment from a time trend; DID without matching relies on parallel trends between dissimilar groups. The 
  combination is what notebook 08 calls "Doubly Robust DID."
  ─────────────────────────────────────────────────                                            
"""