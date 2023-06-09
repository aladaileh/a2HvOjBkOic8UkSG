def re_ranking(starred_candidate_index,data):
    ranked_candidates = data.sort_values(by='similarity', ascending=False)
    starred_candidate_fitness = ranked_candidates['similarity'][starred_candidate_index]
    ranked_candidates['similarity'] = ranked_candidates['similarity'].apply(lambda x: x + .1 if x == ranked_candidates['similarity'][starred_candidate_index] else x)
    re_ranked_candidates = ranked_candidates.sort_values(by='similarity', ascending=False)
    return re_ranked_candidates