# UFC Ranking Columns Removal Analysis

## Executive Summary

After analyzing the missing ranking columns in the UFC dataset, we can safely remove **28 out of 29 ranking columns** that have >50% missing values. This will significantly improve data quality while preserving the useful `BetterRank` column.

## Missing Ranking Columns Analysis

### Ranking Columns to Remove (28 columns)

All of these columns have **97-100% missing values** and are essentially unusable:

#### Weight Class Rankings (24 columns)
- **Red Fighter Rankings**: RWFlyweightRank, RWFeatherweightRank, RWStrawweightRank, RWBantamweightRank, RHeavyweightRank, RLightHeavyweightRank, RMiddleweightRank, RWelterweightRank, RLightweightRank, RFeatherweightRank, RBantamweightRank, RFlyweightRank
- **Blue Fighter Rankings**: BWFlyweightRank, BWFeatherweightRank, BWStrawweightRank, BWBantamweightRank, BHeavyweightRank, BLightHeavyweightRank, BMiddleweightRank, BWelterweightRank, BLightweightRank, BFeatherweightRank, BBantamweightRank, BFlyweightRank

#### Pound-for-Pound Rankings (2 columns)
- **RPFPRank**: 96.1% missing
- **BPFPRank**: 99.0% missing

#### Match Weight Class Rankings (2 columns)
- **RMatchWCRank**: 72.7% missing
- **BMatchWCRank**: 81.6% missing

### Ranking Column to Keep (1 column)

#### BetterRank
- **Missing Values**: 0% (complete data)
- **Unique Values**: 
  - 'neither': 4,684 fights (71.7%)
  - 'Red': 1,751 fights (26.8%)
  - 'Blue': 93 fights (1.4%)
- **Usefulness**: Indicates which fighter has the better ranking in the matchup

## Impact of Removal

### Dataset Size Reduction
- **Original Columns**: 118
- **Columns Removed**: 28
- **Remaining Columns**: 90
- **Data Reduction**: 23.7%

### Data Quality Improvement
- **Eliminates**: 28 columns with 97-100% missing values
- **Preserves**: All complete and usable data
- **Maintains**: Core fight information and fighter statistics

## Remaining Data Quality After Removal

### Excellent Quality Columns (0% missing - 63 columns)
Core fight information and physical attributes:
- **Fight Details**: RedFighter, BlueFighter, Date, Location, Country, Winner, TitleBout, WeightClass, Gender, NumberOfRounds
- **Physical Attributes**: RedAge, BlueAge, RedHeightCms, BlueHeightCms, RedReachCms, BlueReachCms, RedWeightLbs, BlueWeightLbs
- **Career Statistics**: RedWins, BlueWins, RedLosses, BlueLosses, RedDraws, BlueDraws
- **Performance Metrics**: Various career averages and win methods
- **Fight Context**: BetterRank, Finish, FinishDetails

### Acceptable Quality Columns (<5% missing - 6 columns)
- **Betting Odds**: RedOdds, BlueOdds, RedExpectedValue, BlueExpectedValue
- **Fighter Stance**: BlueStance
- **Fight Outcome**: Finish

### Moderate Quality Columns (5-20% missing - 19 columns)
- **Fight Timing**: FinishRound, FinishRoundTime, TotalFightTimeSecs
- **Betting Odds**: RedDecOdds, BlueDecOdds, RSubOdds, BSubOdds, RKOOdds, BKOOdds
- **Performance Metrics**: Various career averages (strikes, takedowns, submissions)

### High Missing Columns (>20% missing - 2 columns)
- **FinishDetails**: 55.7% missing (fight ending details)
- **EmptyArena**: 22.8% missing (COVID-19 era indicator)

## Recommendations

### Immediate Action
1. **Remove 28 ranking columns** with >50% missing values
2. **Keep BetterRank column** as it provides useful ranking information
3. **Proceed with data cleaning** using the 90 remaining columns

### Feature Engineering Opportunities
1. **Use BetterRank**: Create features based on ranking advantage
2. **Weight Class Analysis**: Use WeightClass for division-specific modeling
3. **Physical Advantages**: Calculate height, reach, and weight differentials
4. **Career Trajectories**: Use complete career statistics for fighter analysis

### Data Cleaning Strategy
1. **Remove FinishDetails**: 55.7% missing makes it unreliable
2. **Keep EmptyArena**: Useful for COVID-19 era analysis
3. **Impute betting odds**: 15-20% missing can be handled with imputation
4. **Use performance metrics**: 5-15% missing is manageable

## Benefits of Removal

### Data Quality
- **Eliminates noise**: Removes 28 columns with essentially no data
- **Improves model performance**: Focuses on features with actual data
- **Reduces complexity**: Simpler dataset for modeling

### Computational Efficiency
- **Faster processing**: 23.7% fewer columns to process
- **Reduced memory usage**: Smaller dataset size
- **Cleaner analysis**: Focus on meaningful features

### Modeling Benefits
- **Better feature selection**: Only meaningful features remain
- **Improved interpretability**: Clearer model insights
- **Reduced overfitting**: Less noise in the dataset

## Next Steps

1. **Implement removal**: Remove the 28 ranking columns in data cleaning
2. **Update data quality assessment**: Reassess missing values after removal
3. **Proceed with feature engineering**: Focus on the 90 remaining columns
4. **Validate data quality**: Ensure removal doesn't impact core functionality

## Conclusion

Removing the 28 missing ranking columns is a **highly recommended action** that will:
- Improve data quality significantly
- Reduce dataset complexity by 23.7%
- Focus modeling efforts on meaningful features
- Preserve the useful BetterRank information

This removal addresses the major data quality issue identified in Phase 1.2 and positions the dataset for successful UFC prediction modeling.
