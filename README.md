- Reddit Analysis - created embedding cluster based labels
- Zero shot classification - creates zero shot based labels using llama 3 8b
- few shot zero shot labels - uses the zero shot labels + prompt used in zero shot classification to train llama 3 8b
- Create Synthetic DMs - uses gpt 4o to apply style transfer to Reddit posts to create synthetic DMs.

Everything else is not being actively used
- the scripts in the llama finetuning folder were my attempt at finetuning with all of the examples using a binary classification head. These did not perform as well compared to using few shot instruction finetuning (the few shot scripts)
