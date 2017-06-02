Similarity Measure
------------------
- *Intraclass Correlation (ICC) is applicable to coninuous data and returns a value in interval [-1,1], where 1 indicates exact similarity.*
- Cohen's Kappa is only for nominal vectors
- Weighted Kappa as in http://www.agreestat.com/book3/bookexcerpts/chapter3.pdf could be used.


Average maximum similarity of a data generation method
------------------------------------------------------
1. N times do:
	1. Generate synthetic sample X from parents A (and B if applicable)
	2. Calculate similarity between X and A
	3. If there is one parent:
		1. Return the similarity of X and A
	4. If there are two parents:
		1. Calculate similarity between X and B
		2. Return the maximum of both similarities
2. Return average of loop's returned similarities

This measure could be seen as how "original" the synthetic samples of a data generator are. A value of 1 corresponds to random oversampling, where generated data points are exact copies of their parent. If two parents are very close, the measure will also tent towards 1. Strongly differing samples will have a value much less than 1, potentially less than 0. A very low value can also hint towards bad synthetic sample quality, and shouldn't be used to judge the method as "good", but only as "diverse".