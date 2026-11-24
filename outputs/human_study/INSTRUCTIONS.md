# Rating instructions

About 15 minutes. Do it alone, in one sitting, without looking up any name, and without
talking about the names with the others until all three of us are done. Do not open the
`key_*.csv` files (they say where each name came from).

## The easy way: the rating app

Open **https://bloohunnits.github.io/723finalproject/** on your phone or laptop. It is an ordinary web page: no account, no
install. (The same page is `rating_app.html` in this folder, if you would rather open the file.)

1. Type your first name and press Start.
2. Part 1: 120 names, one at a time. Tap 1 to 5 (or press the number keys). Ten of them
   come back a second time later on. That is on purpose, just answer again.
3. Part 2: 60 pairs. Tap Same name or Different name (or press Y / N).
4. On the last screen press **Send my ratings** (phones) or **Copy my ratings**, and send that
   whole block of text to Andre.

You can go back to change an answer, and you can close the tab and resume later on the same
device. Nothing is sent anywhere automatically.

### What you are answering

**Part 1.** Could this be the given name of a character in a fantasy role-playing game such as Dungeons & Dragons?

5 = Yes. Reads as a finished name. You would not look twice at it on a character sheet.
4 = Probably. Sounds like a name, with something slightly off.
3 = Borderline. Pronounceable, but it is unclear whether it is a name.
2 = Probably not. Awkward letter sequences, or it reads as a fragment.
1 = No. Unpronounceable or random letters.

Names are lowercase and may contain apostrophes or hyphens. Judge only whether the name
works as a name. Do not reward a name for being familiar, and do not penalize it for
being unusual.

**Part 2.** Is the first name basically the second name: a respelling, or the same name with a small addition or change?

Examples: `jonn` / `jon` is the same name. `kaius` / `caius` is the same name. `torvus` /
`corvus` is different, a different first letter makes a different name.

## The manual way: CSV sheets

Only if the app does not work for you. Fill the `rating` column of `sheet_A.csv` with 1 to 5
and the `same` column of `sheet_B.csv` with 1 or 0, using the two questions above. Save your
copies in this folder as `rater_<yourname>_A.csv` and `rater_<yourname>_B.csv`, and keep the
`id` column as it is.

## For Andre: turning app results into files

Save each person's pasted text to a file and run
`python scripts/human_tasks.py import <file>`. It writes `rater_<name>_A.csv`,
`rater_<name>_B.csv` and `rater_<name>_repeats.csv` here and refuses incomplete results.
