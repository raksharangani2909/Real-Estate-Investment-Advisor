# EDA findings

Dataset: 250,000 cleaned properties, 20 states, 42 cities. One chart per question is in `reports/figures/`.

> **Read this first:** the original features are almost perfectly independent of each other and of price (pairwise |r| < 0.01). That is typical of a synthetic/randomly generated dataset, so most 'no relationship' answers below are genuine findings about THIS data, not mistakes.

### Q1. Distribution of property prices
Prices run 10-500 lakhs, mean 254.6, median 253.9, std 141.3. The histogram is essentially flat (skew 0.01), i.e. close to uniform - there is no long luxury tail.

### Q2. Distribution of property sizes
Sizes run 500-5000 sq ft, mean 2750, median 2747; the shape is flat/uniform (skew 0.00).

### Q3. Price per sq ft by property type
Median INR/sq ft: Apartment 9,245, Independent House 9,305, Villa 9,190. Spread between types is only 1.3% - property type barely moves price/sq ft.

### Q4. Relationship between property size and price
Pearson r = -0.003 (negligible). In this dataset a bigger home is NOT priced higher in absolute terms - size and price vary independently, so price per sq ft falls as size rises (r with size = -0.61).

### Q5. Outliers in price per sq ft or property size
IQR rule: price/sq ft has 19,723 outliers (7.9%), all on the HIGH side (small flats with high prices, max 99,182 INR/sq ft). Size (0) and price (0) have none. These are valid records, so they are flagged (PPSF_Outlier) rather than deleted; models use trees/scaling that tolerate them.

### Q6. Average price per sq ft by state
Highest: Karnataka (13,252 INR), lowest: Punjab (12,931 INR) - only a 2.5% gap across 20 states.

### Q7. Average property price by city
Highest: Bangalore (258.5 lakhs), lowest: Cuttack (250.8 lakhs); all 42 cities sit within 3.1% of each other (city mean of means 254.7).

### Q8. Median age of properties by locality
Locality names repeat in every city, so a locality = (City, Locality). Across 20,999 localities the median age ranges 3-34 years (overall median 18.5). Each locality only has ~12 listings, so individual values are noisy. Oldest: Mysore / Locality_444 (34 yrs).

### Q9. How BHK is distributed across cities
Every city has all five BHK sizes at roughly 20% each (range 18.6%-22.0%) - no city specialises in small or large homes.

### Q10. Price trends for the top 5 most expensive localities
Top 5 (localities with >=8 listings, ranked by mean price): Hyderabad / Locality_177 (403 lakhs); Jaipur / Locality_213 (400 lakhs); Pune / Locality_253 (397 lakhs); Patna / Locality_116 (396 lakhs); Bhubaneswar / Locality_362 (396 lakhs). The dataset has no transaction date, so 'trend' = average price by construction period. Lines are jagged (few listings per band) and show no consistent rise for newer homes.

### Q11. How numeric features are correlated
Strongest pairs: Transport_Score vs Infrastructure_Score r=0.71; Size_in_SqFt vs Price_per_SqFt_INR r=-0.61; Floor_No vs Total_Floors r=0.58; Price_in_Lakhs vs Price_per_SqFt_INR r=0.56; Nearby_Schools vs Infrastructure_Score r=0.50. Every strong pair involves a derived field: price/sq ft comes from price and size, infrastructure score from schools, hospitals and transport, and Floor_No vs Total_Floors is high only because Floor_No was capped at Total_Floors during cleaning. Among the ORIGINAL columns in the raw file, every pairwise correlation is below 0.01 (checked on the uncleaned data).

### Q12. How nearby schools relate to price per sq ft
Pearson r with price/sq ft = -0.000 (negligible); mean price/sq ft ranges 12,888-13,292 INR across levels - no systematic trend. (Correlation with total price: 0.000.)

### Q13. How nearby hospitals relate to price per sq ft
Pearson r with price/sq ft = -0.000 (negligible); mean price/sq ft ranges 12,917-13,257 INR across levels - no systematic trend. (Correlation with total price: -0.003.)

### Q14. How price varies by furnished status
Avg price (lakhs): Unfurnished 255.0, Semi-furnished 254.3, Furnished 254.4. Spread 0.3% - furnishing carries no price premium here.

### Q15. How price per sq ft varies by facing direction
Mean INR/sq ft: East 13,024, North 13,025, South 13,044, West 13,140. Gap max-min = 0.9% - direction does not matter in this data.

### Q16. How many properties belong to each owner type
Broker: 83,479, Owner: 83,268, Builder: 83,253 (about one-third each).

### Q17. How many properties are available under each availability status
Under_Construction: 125,035, Ready_to_Move: 124,965. (The data has only these two statuses - no 'Sold'.)

### Q18. Does parking space affect property price
Avg price with parking 254.7 vs without 254.4 lakhs (+0.12%). No meaningful effect on price. (Parking does add to the modelled 5-year growth rule.)

### Q19. How amenities affect price per sq ft
Amenity count vs price/sq ft r = 0.002 (negligible). Per-amenity price/sq ft difference (has vs hasn't): Clubhouse +0.2%, Gym +0.8%, Garden +0.1%, Pool +0.3%, Playground -0.2%. Mixed signs of small size = noise, not a premium.

### Q20. How public transport accessibility relates to price per sq ft or investment potential
Price/sq ft is flat across Low/Medium/High (Low 13,086, Medium 13,062, High 13,027 INR). Investment potential, however, rises sharply because transport is a driver of the growth rule: share of Good Investments = Low 9%, Medium 31%, High 52% (by construction of the rule, not observed market behaviour).

### X1. Correlation between area and investment return
Size vs annual growth r = 0.194; Good-Investment share by size band: 500-1k 8%, 1-2k 16%, 2-3k 29%, 3-4k 40%, 4-5k 51%. The share of Good Investments rises steeply with size because larger homes have a lower price/sq ft (see Q4): they more often qualify for the 'undervalued' growth bonus and pass the 'not over-priced' test. This is an effect of the investment rule, not observed market returns.

### X2. Impact of crime rate on 'Good Investment' classification
The dataset has NO crime-rate column, so the Security flag is used as a safety proxy: Good-Investment share 33.8% with security vs 28.3% without. The gap exists only because the growth rule gives security a small bonus.

### X3. Relationship between infrastructure score and resale value
Infrastructure score vs today's price r = -0.001 and vs price/sq ft r = -0.001 (both negligible). Vs the projected 5-year value r = 0.072: infrastructure raises the FORECAST value only because it drives the growth rule.
