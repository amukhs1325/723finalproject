# Rating instructions

About 20 minutes. Please do both sheets alone, in one sitting, without looking up any name
and without opening the `key_*.csv` files (they say where each name came from).

## Sheet A: `sheet_A.csv` (120 names)

Fill the `rating` column with a number from 1 to 5.

Could this be the given name of a character in a fantasy role-playing game such as Dungeons & Dragons?

5 = Yes. Reads as a finished name. You would not look twice at it on a character sheet.
4 = Probably. Sounds like a name, with something slightly off.
3 = Borderline. Pronounceable, but it is unclear whether it is a name.
2 = Probably not. Awkward letter sequences, or it reads as a fragment.
1 = No. Unpronounceable or random letters.

Names are lowercase and may contain apostrophes or hyphens. Judge only whether the name
works as a name. Do not reward a name for being familiar, and do not penalize it for
being unusual.

## Sheet B: `sheet_B.csv` (60 pairs)

Fill the `same` column with 1 or 0.

Is the LEFT name basically the RIGHT name: a respelling, or the same name with a small addition or change? 1 = yes, 0 = no.

Examples: `jonn` / `jon` is 1. `kaius` / `caius` is 1. `torvus` / `corvus` is 0, a
different first letter makes a different name.

## Saving

Save your copies in this folder as `rater_<yourname>_A.csv` and `rater_<yourname>_B.csv`.
Keep the `id` column as it is.
