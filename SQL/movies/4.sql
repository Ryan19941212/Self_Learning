-- Determine the number of movies with an IMDb rating of 10.0
SELECT count(*)
FROM ratings
WHERE rating >= 10;
