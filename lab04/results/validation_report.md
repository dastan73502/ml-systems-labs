# Validation report

Rows in: 48842  |  Rows out: 48842  |  Rows dropped: 0
Checks run: 45  |  Failed: 0

| check | column | rule | violations | status | action | example rows |
|---|---|---|---:|---|---|---|
| columns | * | exact schema columns | 0 | PASS | missing=[] extra=[] | [] |
| type | age | numeric | 0 | PASS | none | [] |
| type | age | integer-valued | 0 | PASS | none | [] |
| missing | age | no nulls | 0 | PASS | none | [] |
| range | age | 17 <= x <= 90 | 0 | PASS | none | [] |
| category | workclass | in 9 allowed values | 0 | PASS | none | [] |
| missing | workclass | '?' allowed, counted | 2799 | INFO | kept, imputed after split | [4, 6, 13, 22, 35] |
| type | fnlwgt | numeric | 0 | PASS | none | [] |
| type | fnlwgt | integer-valued | 0 | PASS | none | [] |
| missing | fnlwgt | no nulls | 0 | PASS | none | [] |
| category | education | in 17 allowed values | 0 | PASS | none | [] |
| missing | education | '?' allowed, counted | 0 | INFO | kept, imputed after split | [] |
| type | education-num | numeric | 0 | PASS | none | [] |
| type | education-num | integer-valued | 0 | PASS | none | [] |
| missing | education-num | no nulls | 0 | PASS | none | [] |
| range | education-num | 1 <= x <= 16 | 0 | PASS | none | [] |
| category | marital-status | in 8 allowed values | 0 | PASS | none | [] |
| missing | marital-status | '?' allowed, counted | 0 | INFO | kept, imputed after split | [] |
| category | occupation | in 15 allowed values | 0 | PASS | none | [] |
| missing | occupation | '?' allowed, counted | 2809 | INFO | kept, imputed after split | [4, 6, 13, 22, 35] |
| category | relationship | in 7 allowed values | 0 | PASS | none | [] |
| missing | relationship | '?' allowed, counted | 0 | INFO | kept, imputed after split | [] |
| category | race | in 6 allowed values | 0 | PASS | none | [] |
| missing | race | '?' allowed, counted | 0 | INFO | kept, imputed after split | [] |
| category | sex | in 3 allowed values | 0 | PASS | none | [] |
| missing | sex | '?' allowed, counted | 0 | INFO | kept, imputed after split | [] |
| type | capital-gain | numeric | 0 | PASS | none | [] |
| type | capital-gain | integer-valued | 0 | PASS | none | [] |
| missing | capital-gain | no nulls | 0 | PASS | none | [] |
| range | capital-gain | 0 <= x <= 99999 | 0 | PASS | none | [] |
| type | capital-loss | numeric | 0 | PASS | none | [] |
| type | capital-loss | integer-valued | 0 | PASS | none | [] |
| missing | capital-loss | no nulls | 0 | PASS | none | [] |
| range | capital-loss | 0 <= x <= 4356 | 0 | PASS | none | [] |
| type | hours-per-week | numeric | 0 | PASS | none | [] |
| type | hours-per-week | integer-valued | 0 | PASS | none | [] |
| missing | hours-per-week | no nulls | 0 | PASS | none | [] |
| range | hours-per-week | 1 <= x <= 99 | 0 | PASS | none | [] |
| category | native-country | in 42 allowed values | 0 | PASS | none | [] |
| missing | native-country | '?' allowed, counted | 857 | INFO | kept, imputed after split | [19, 65, 83, 188, 253] |
| category | income | in 2 allowed values | 0 | PASS | none | [] |
| missing | income | no missing label | 0 | PASS | none | [] |
| consistency | education/education-num | education-num matches education | 0 | PASS | none | [] |
| duplicates | * | exact duplicate rows (all 15 columns) | 52 | INFO | kept in raw; dropped from train split only | [864, 11189, 11212, 13848, 15960] |
| duplicates | * | duplicate feature rows (14 features) | 57 | INFO | see label audit for conflicting labels | [864, 11189, 11212, 13848, 15960] |
