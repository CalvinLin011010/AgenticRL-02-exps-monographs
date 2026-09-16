# Step-100 token signals

This is a small teacher-forced re-scoring pass over six historical action segments. It is not a replay of the original sampling and cannot recover the original policy entropy. Each token stores full-vocabulary entropy, actual-token logprob/probability, token id/text, and coverage before/after the selected action.

| q | event | coverage before -> after | tokens | mean entropy | min | max | variance | mean logprob |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 0.00 -> 1.00 | 42 | 0.0363 | 0.0000 | 0.4285 | 0.0105 | -0.0634 |
| 7 | 3 | 0.00 -> 1.00 | 42 | 0.1468 | 0.0000 | 1.2174 | 0.0768 | -0.5623 |
| 15 | 4 | 0.00 -> 1.00 | 40 | 0.0542 | 0.0000 | 0.6314 | 0.0179 | -1.7519 |
| 16 | 2 | 0.00 -> 1.00 | 33 | 0.0972 | 0.0000 | 1.0314 | 0.0602 | -0.3553 |
| 23 | 3 | 0.00 -> 1.00 | 41 | 0.0451 | 0.0000 | 0.7011 | 0.0206 | -0.5653 |
| 26 | 2 | 0.00 -> 1.00 | 56 | 0.0731 | 0.0000 | 0.9333 | 0.0378 | -0.2565 |

`token_entropy_heatmap.svg` is hoverable in a browser. Colors encode full-vocabulary entropy, not lexical coverage.
