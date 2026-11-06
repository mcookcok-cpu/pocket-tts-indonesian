# Voice prompts

Four Indonesian voice prompts, four different speakers, 48 kHz mono, ~4.7 s
each. Use any of them with `--voice`, or bring your own recording.

From [Common Voice 22.0](https://commonvoice.mozilla.org/) Indonesian, which is
**CC0** — public domain, free to redistribute and use, no attribution required.

These four are not arbitrary picks. In a cross-similarity test, generations made
from them sit closer to their own prompt than to each other, which is what you
want and is not guaranteed:

| | cosine |
|---|---|
| generation against its own prompt | 0.892 |
| generation against a different prompt | 0.775 |
| two generations from different prompts | 0.771 |
| the four prompts against each other | 0.789 |

For contrast, prompts taken from 16 kHz YouTube audio produced generations
scoring 0.939 against each other — indistinguishable voices, whatever prompt
they started from. Prompt bandwidth is the single biggest lever on how
distinctly a voice clones, so a quiet 48 kHz recording beats a louder
compressed one.
