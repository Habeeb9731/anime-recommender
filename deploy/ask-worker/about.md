# About AnimeRec

AnimeRec (anime.abdulhabeeb.com) is a top-10 anime recommender demo. Visitors answer a short onboarding (genres they like, titles they loved, genres to skip, series or movies, how recent), and get ten picks, each with a one-line reason. They can change their picks later with the search button and refine with the plus button (genre, release year, new titles only, sequels).

How to use the site:
- The search button (magnifier) adds titles you loved or disliked; the recommendations update straight away.
- The plus button filters by genre, release year, new titles only, and whether sequels of your picks are included. "Retake preferences" starts onboarding again.
- The rail on the left (or the pill at the bottom on phones) switches between All, TV, Movie, OVA, ONA and Special.
- "Love this" on the featured title adds it to your picks.
- Answers are saved only in the visitor's browser (local storage and the page URL). Nothing is stored on a server.

The catalog: 10,720 titles from MyAnimeList, with activity up to February 2018, so anything that first aired after early 2018 is not included. Adult titles and music videos are excluded. Star ratings are MyAnimeList community scores, shown for reference only and never used by the recommender.

Built by Abdul Habeeb, a Data Science and Full-Stack Developer and Master's student in Data Science and AI at Saarland University in Saarbrücken, Germany. Portfolio: abdulhabeeb.com. The source code is public on GitHub (Habeeb9731/anime-recommender), and there is a case study on his portfolio.

This assistant runs on Claude Haiku in a Cloudflare Worker. It suggests titles from its own knowledge, and every suggestion is checked against the AnimeRec catalog before it is shown, so only titles the demo knows appear. Adding them to your picks hands them to the recommender itself, which then ranks the full catalog for you.
