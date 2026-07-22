# Heuristic Summary

_Generated: 2026-02-11T21:25:58.666758Z_

## Context
- **model**: decision_tree
- **data**: data/with_features.jsonl
- **metrics**: {'accuracy': 0.5873983739837398, 'f1': 0.13872731960540902, 'precision': 0.23845132914826012, 'recall': 0.20184935246708502}

## Top Heuristics
1. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_location <= 0.5000 THEN class=14  
   - support=0.000 | precision=0.004
2. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_location > 0.5000 THEN class=14  
   - support=0.000 | precision=0.000
3. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_video_surveillance > 0.5000 THEN class=14  
   - support=0.000 | precision=0.000
4. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 AND kw_video_surveillance <= 0.5000 THEN class=14  
   - support=0.000 | precision=0.000
5. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 AND kw_video_surveillance > 0.5000 THEN class=13  
   - support=0.000 | precision=0.000
6. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric > 0.5000 THEN class=8  
   - support=0.000 | precision=0.000
7. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 THEN class=9  
   - support=0.000 | precision=0.000
8. (rule) IF kw_privacy <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric > 0.5000 THEN class=8  
   - support=0.000 | precision=0.000
9. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location <= 0.5000 AND kw_biometric <= 0.5000 THEN class=3  
   - support=0.000 | precision=0.000
10. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location <= 0.5000 AND kw_biometric > 0.5000 THEN class=0  
   - support=0.000 | precision=0.000
11. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location > 0.5000 THEN class=15  
   - support=0.000 | precision=0.000
12. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement > 0.5000 THEN class=3  
   - support=0.000 | precision=0.000
13. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 THEN class=15  
   - support=0.000 | precision=0.000
14. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 THEN class=6  
   - support=0.000 | precision=0.000
15. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric > 0.5000 THEN class=10  
   - support=0.000 | precision=0.000
16. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location <= 0.5000 THEN class=15  
   - support=0.000 | precision=0.000
17. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location > 0.5000 THEN class=15  
   - support=0.000 | precision=0.000
18. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 THEN class=0  
   - support=0.000 | precision=0.000
19. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric > 0.5000 THEN class=15  
   - support=0.000 | precision=0.000
20. (rule) IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 AND kw_location <= 0.5000 AND kw_reg_enforcement <= 0.5000 THEN class=6  
   - support=0.000 | precision=0.000

## Appendix: Heuristics Table

| # | Kind | Text | Support | Precision |
|:-:|:-----|:-----|:-------:|:---------:|
| 1 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_location <= 0.5000 THEN class=14 | 0.000 | 0.004 |
| 2 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_location > 0.5000 THEN class=14 | 0.000 | 0.000 |
| 3 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_video_surveillance > 0.5000 THEN class=14 | 0.000 | 0.000 |
| 4 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 AND kw_video_surveillance <= 0.5000 THEN class=14 | 0.000 | 0.000 |
| 5 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 AND kw_video_surveillance > 0.5000 THEN class=13 | 0.000 | 0.000 |
| 6 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric > 0.5000 THEN class=8 | 0.000 | 0.000 |
| 7 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 THEN class=9 | 0.000 | 0.000 |
| 8 | rule | IF kw_privacy <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric > 0.5000 THEN class=8 | 0.000 | 0.000 |
| 9 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location <= 0.5000 AND kw_biometric <= 0.5000 THEN class=3 | 0.000 | 0.000 |
| 10 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location <= 0.5000 AND kw_biometric > 0.5000 THEN class=0 | 0.000 | 0.000 |
| 11 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location > 0.5000 THEN class=15 | 0.000 | 0.000 |
| 12 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_reg_enforcement > 0.5000 THEN class=3 | 0.000 | 0.000 |
| 13 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 THEN class=15 | 0.000 | 0.000 |
| 14 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 THEN class=6 | 0.000 | 0.000 |
| 15 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance <= 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric > 0.5000 THEN class=10 | 0.000 | 0.000 |
| 16 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location <= 0.5000 THEN class=15 | 0.000 | 0.000 |
| 17 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement <= 0.5000 AND kw_location > 0.5000 THEN class=15 | 0.000 | 0.000 |
| 18 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric <= 0.5000 AND kw_reg_enforcement > 0.5000 THEN class=0 | 0.000 | 0.000 |
| 19 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty <= 0.5000 AND kw_biometric > 0.5000 THEN class=15 | 0.000 | 0.000 |
| 20 | rule | IF kw_privacy > 0.5000 AND kw_video_surveillance > 0.5000 AND kw_monetary_penalty > 0.5000 AND kw_biometric <= 0.5000 AND kw_location <= 0.5000 AND kw_reg_enforcement <= 0.5000 THEN class=6 | 0.000 | 0.000 |
