"""
UFC ELO Rating System - Exact replica of tennis ELO system
Implements the same ELO logic used in the tennis model for UFC fights
"""

def createUFCStats():
    """
    Create initial stats dictionary - EXACT SAME as tennis createStats()
    
    Returns:
        dict: Dictionary containing all UFC stats structures
    """
    import numpy as np
    from collections import defaultdict, deque

    prev_stats = {}

    # EXACT SAME structure as tennis model
    prev_stats["elo_fighters"] = defaultdict(int)  # Base ELO ratings
    prev_stats["elo_weightclass_fighters"] = defaultdict(lambda: defaultdict(int))  # Weight class specific ELO
    prev_stats["elo_grad_fighters"] = defaultdict(lambda: deque(maxlen=1000))  # ELO gradient tracking
    prev_stats["last_k_fights"] = defaultdict(lambda: deque(maxlen=1000))  # Last K fight results
    prev_stats["last_k_fights_stats"] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=1000)))  # Fight statistics
    prev_stats["fights_played"] = defaultdict(int)  # Total fights
    prev_stats["h2h"] = defaultdict(int)  # Head-to-head records
    prev_stats["h2h_weightclass"] = defaultdict(lambda: defaultdict(int))  # H2H by weight class

    return prev_stats


def updateUFCStats(fight, prev_stats):
    """
    Update UFC stats after each fight - EXACT SAME logic as tennis updateStats()
    
    Args:
        fight: Row from UFC dataset containing fight information
        prev_stats: Current stats dictionary
    
    Returns:
        dict: Updated stats dictionary
    """
    from utils.ufc_common import getWinnerLoserIDS
    import numpy as np

    # Get Winner and Loser IDs (EXACT SAME as tennis)
    red_id = fight.RedFighterID
    blue_id = fight.BlueFighterID
    weight_class = fight.WeightClass
    result = fight.RESULT
    
    w_id, l_id = getWinnerLoserIDS(red_id, blue_id, result)

    ######################## UPDATE ########################
    ############## ELO ##############
    # Get current ELO ratings (BEFORE this fight) - EXACT SAME as tennis
    elo_w = prev_stats["elo_fighters"].get(w_id, 1500)  # Default 1500
    elo_l = prev_stats["elo_fighters"].get(l_id, 1500)
    elo_weightclass_w = prev_stats["elo_weightclass_fighters"][weight_class].get(w_id, 1500)
    elo_weightclass_l = prev_stats["elo_weightclass_fighters"][weight_class].get(l_id, 1500)

    # Calculate expected probabilities - EXACT SAME formula as tennis
    k = 24  # EXACT SAME K-factor
    exp_w = 1/(1+(10**((elo_l-elo_w)/400)))
    exp_l = 1/(1+(10**((elo_w-elo_l)/400)))
    exp_weightclass_w = 1/(1+(10**((elo_weightclass_l-elo_weightclass_w)/400)))
    exp_weightclass_l = 1/(1+(10**((elo_weightclass_w-elo_weightclass_l)/400)))

    # Update ELO ratings for next fight - EXACT SAME as tennis
    elo_w += k*(1-exp_w)
    elo_l += k*(0-exp_l)
    elo_weightclass_w += k*(1-exp_weightclass_w)
    elo_weightclass_l += k*(0-exp_weightclass_l)

    # Store updated ratings
    prev_stats["elo_fighters"][w_id] = elo_w
    prev_stats["elo_fighters"][l_id] = elo_l
    prev_stats["elo_weightclass_fighters"][weight_class][w_id] = elo_weightclass_w
    prev_stats["elo_weightclass_fighters"][weight_class][l_id] = elo_weightclass_l

    ########################################################
    ############## ELO GRAD ##############
    prev_stats["elo_grad_fighters"][w_id].append(elo_w)
    prev_stats["elo_grad_fighters"][l_id].append(elo_l)
    
    ########################################################
    ############## Fights Played ##############
    prev_stats["fights_played"][w_id] += 1
    prev_stats["fights_played"][l_id] += 1

    ########################################################
    ############## Last K Fights Won ##############
    prev_stats["last_k_fights"][w_id].append(1)
    prev_stats["last_k_fights"][l_id].append(0)

    ########################################################
    ############# H2H and H2H by Weight Class #############
    prev_stats["h2h"][(w_id, l_id)] += 1
    prev_stats["h2h_weightclass"][weight_class][(w_id, l_id)] += 1

    ########################################################
    ############# UPDATE UFC-Specific Statistics #############
    # Extract fight statistics for tracking
    if hasattr(fight, 'RedStrLanded') and hasattr(fight, 'RedStrAttempted'):
        # Red fighter statistics
        if fight.RedStrAttempted > 0:
            red_strike_acc = (fight.RedStrLanded / fight.RedStrAttempted) * 100
        else:
            red_strike_acc = 0
            
        if hasattr(fight, 'RedTDLanded') and hasattr(fight, 'RedTDAttempted'):
            if fight.RedTDAttempted > 0:
                red_td_acc = (fight.RedTDLanded / fight.RedTDAttempted) * 100
            else:
                red_td_acc = 0
        else:
            red_td_acc = 0

        # Blue fighter statistics  
        if fight.BlueStrAttempted > 0:
            blue_strike_acc = (fight.BlueStrLanded / fight.BlueStrAttempted) * 100
        else:
            blue_strike_acc = 0
            
        if hasattr(fight, 'BlueTDLanded') and hasattr(fight, 'BlueTDAttempted'):
            if fight.BlueTDAttempted > 0:
                blue_td_acc = (fight.BlueTDLanded / fight.BlueTDAttempted) * 100
            else:
                blue_td_acc = 0
        else:
            blue_td_acc = 0

        # Store winner and loser stats
        if red_id == w_id:
            w_strike_acc, l_strike_acc = red_strike_acc, blue_strike_acc
            w_td_acc, l_td_acc = red_td_acc, blue_td_acc
        else:
            w_strike_acc, l_strike_acc = blue_strike_acc, red_strike_acc
            w_td_acc, l_td_acc = blue_td_acc, red_td_acc

        # Update fight statistics
        prev_stats["last_k_fights_stats"][w_id]["strike_acc"].append(w_strike_acc)
        prev_stats["last_k_fights_stats"][l_id]["strike_acc"].append(l_strike_acc)
        prev_stats["last_k_fights_stats"][w_id]["td_acc"].append(w_td_acc)
        prev_stats["last_k_fights_stats"][l_id]["td_acc"].append(l_td_acc)

    # Track KO and submission rates
    if hasattr(fight, 'FinishRound'):
        # Determine if fight ended in KO/TKO or submission
        is_ko = False
        is_sub = False
        
        if hasattr(fight, 'Method'):
            method = str(fight.Method).lower()
            if 'ko' in method or 'tko' in method or 'knockout' in method:
                is_ko = True
            elif 'submission' in method or 'sub' in method:
                is_sub = True

        # Update rates for winner and loser
        prev_stats["last_k_fights_stats"][w_id]["ko_rate"].append(1 if is_ko else 0)
        prev_stats["last_k_fights_stats"][l_id]["ko_rate"].append(0)
        prev_stats["last_k_fights_stats"][w_id]["sub_rate"].append(1 if is_sub else 0)
        prev_stats["last_k_fights_stats"][l_id]["sub_rate"].append(0)
    
    return prev_stats


def getUFCStats(fighter1, fighter2, fight, prev_stats):
    """
    Get UFC stats for prediction - EXACT SAME structure as tennis getStats()
    
    Args:
        fighter1: Fighter 1 dictionary with ID, Age, Height, etc.
        fighter2: Fighter 2 dictionary with ID, Age, Height, etc.
        fight: Fight context dictionary (WeightClass, NumberOfRounds, etc.)
        prev_stats: Current stats dictionary
    
    Returns:
        dict: Dictionary with all calculated stats for prediction
    """
    from utils.ufc_common import mean
    import numpy as np

    output = {}
    FIGHTER1_ID = fighter1["ID"]
    FIGHTER2_ID = fighter2["ID"]
    WEIGHT_CLASS = fight["WeightClass"]

    # Get Differences - UFC equivalents of tennis differences
    output["NumberOfRounds"] = fight["NumberOfRounds"]
    output["AGE_DIFF"] = fighter1["AGE"] - fighter2["AGE"]
    output["HEIGHT_DIFF"] = fighter1["HEIGHT"] - fighter2["HEIGHT"]
    output["REACH_DIFF"] = fighter1["REACH"] - fighter2["REACH"]
    output["WEIGHT_DIFF"] = fighter1["WEIGHT"] - fighter2["WEIGHT"]

    # Get Stats from Dictionary - EXACT SAME as tennis
    elo_fighters = prev_stats["elo_fighters"]
    elo_weightclass_fighters = prev_stats["elo_weightclass_fighters"]
    elo_grad_fighters = prev_stats["elo_grad_fighters"]
    last_k_fights = prev_stats["last_k_fights"]
    last_k_fights_stats = prev_stats["last_k_fights_stats"]
    fights_played = prev_stats["fights_played"]
    h2h = prev_stats["h2h"]
    h2h_weightclass = prev_stats["h2h_weightclass"]

    ####################### GET STATS ########################
    # EXACT SAME ELO features as tennis
    output["ELO_DIFF"] = elo_fighters[FIGHTER1_ID] - elo_fighters[FIGHTER2_ID]
    output["ELO_WEIGHTCLASS_DIFF"] = elo_weightclass_fighters[WEIGHT_CLASS][FIGHTER1_ID] - elo_weightclass_fighters[WEIGHT_CLASS][FIGHTER2_ID]
    output["N_FIGHTS_DIFF"] = fights_played[FIGHTER1_ID] - fights_played[FIGHTER2_ID]
    output["H2H_DIFF"] = h2h[(FIGHTER1_ID, FIGHTER2_ID)] - h2h[(FIGHTER2_ID, FIGHTER1_ID)]
    output["H2H_WEIGHTCLASS_DIFF"] = h2h_weightclass[WEIGHT_CLASS][(FIGHTER1_ID, FIGHTER2_ID)] - h2h_weightclass[WEIGHT_CLASS][(FIGHTER2_ID, FIGHTER1_ID)]

    # EXACT SAME K-values as tennis
    for k in [3, 5, 10, 25, 50, 100, 200]:
        ############## Last K Fights Won ##############
        if len(last_k_fights[FIGHTER1_ID]) >= k and len(last_k_fights[FIGHTER2_ID]) >= k:
            output["WIN_LAST_"+str(k)+"_DIFF"] = sum(list(last_k_fights[FIGHTER1_ID])[-k:])-sum(list(last_k_fights[FIGHTER2_ID])[-k:])
        else:
            output["WIN_LAST_"+str(k)+"_DIFF"] = 0
        
        ############## ELO GRAD ##############
        # EXACT SAME gradient calculation as tennis
        if len(elo_grad_fighters[FIGHTER1_ID]) >= k and len(elo_grad_fighters[FIGHTER2_ID]) >= k:
            elo_grad_f1 = list(elo_grad_fighters[FIGHTER1_ID])[-k:]
            elo_grad_f2 = list(elo_grad_fighters[FIGHTER2_ID])[-k:]
            slope_1 = np.polyfit(np.arange(len(elo_grad_f1)), np.array(elo_grad_f1), 1)[0]
            slope_2 = np.polyfit(np.arange(len(elo_grad_f2)), np.array(elo_grad_f2), 1)[0]
            output["ELO_GRAD_LAST_"+str(k)+"_DIFF"] = slope_1-slope_2
        else:
            output["ELO_GRAD_LAST_"+str(k)+"_DIFF"] = 0

        ############# UFC-Specific Statistics #############
        # Replace tennis serve stats with UFC fight stats
        output["STRIKE_ACC_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_fights_stats[FIGHTER1_ID]["strike_acc"])[-k:])-mean(list(last_k_fights_stats[FIGHTER2_ID]["strike_acc"])[-k:])
        output["TD_ACC_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_fights_stats[FIGHTER1_ID]["td_acc"])[-k:])-mean(list(last_k_fights_stats[FIGHTER2_ID]["td_acc"])[-k:])
        output["KO_RATE_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_fights_stats[FIGHTER1_ID]["ko_rate"])[-k:])-mean(list(last_k_fights_stats[FIGHTER2_ID]["ko_rate"])[-k:])
        output["SUB_RATE_LAST_"+str(k)+"_DIFF"] = mean(list(last_k_fights_stats[FIGHTER1_ID]["sub_rate"])[-k:])-mean(list(last_k_fights_stats[FIGHTER2_ID]["sub_rate"])[-k:])

    return output


if __name__ == '__main__':
    # Test the functions
    print("UFC ELO Utils created successfully!")
    
    # Test createUFCStats
    stats = createUFCStats()
    print("Created UFC stats structure with keys:", list(stats.keys()))
    
    # Verify structure matches tennis model
    expected_keys = [
        "elo_fighters", "elo_weightclass_fighters", "elo_grad_fighters",
        "last_k_fights", "last_k_fights_stats", "fights_played", 
        "h2h", "h2h_weightclass"
    ]
    
    missing_keys = [key for key in expected_keys if key not in stats]
    if not missing_keys:
        print("✓ All expected keys present in UFC stats structure")
    else:
        print("✗ Missing keys:", missing_keys)
