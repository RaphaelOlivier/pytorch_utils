def greedy_search(model, keys, values, init_state, init_context, beam_size, max_step, start, stop):
    """
    Assumes model has the following methods:
    get_one_input(word_index:torch.LongTensor,context) --> input
    one_step_decoding(input,state,keys,values) --> decoded, new_context, new_state
    get_word_scores(decoded, context) --> scores
    """
    time_step = 0
    score = torch.FloatTensor([0]).to(start.device)
    tgt_ids = []
    current_word = start
    tgt_ids.append(current_word)
    current_context = init_context
    current_state = init_state

    while (time_step < max_step and current_word[0] != stop[0]):
        time_step += 1
        input = model.get_one_input(current_word, current_context)
        decoded, current_context, current_state = self.one_step_decoding(
            input, current_state, values, keys)
        scores = self.get_word_scores(decoded, current_context)
        current_score, current_word = torch.max(scores, dim=1)
        tgt_ids.append(current_word)
        score += current_score
        if current_word != stop:
            tgt_ids.append(stop)
        score = (score.cpu().detach().numpy() - np.log(time_step)).item()

        return torch.cat(tgt_ids), score


def beam_search(model, keys, values, init_state, init_context, beam_size, max_step, start, stop):
    """
    Assumes model has the following methods:
    get_one_input(word_index:torch.LongTensor,context) --> input
    one_step_decoding(input,state,keys,values) --> decoded, new_context, new_state
    get_word_scores(decoded, context) --> scores
    """
    time_step = 0
    hypotheses = [([start], init_state, init_context, 0)]

    while (time_step < max_step):
        time_step += 1
        next_hypotheses = []
        stopped = True

        for hypothesis in hypotheses:
            current_sentence = hypothesis[0]
            current_word = current_sentence[-1]

            if current_word == stop:
                next_hypotheses.append(hypothesis)
                continue
            else:
                stopped = False

            current_state = hypothesis[1]
            current_context = hypothesis[2]
            current_score = hypothesis[3]

            input = model.get_one_input(current_word, current_context)

            decoded, new_context, current_state = model.one_step_decoding(
                input, current_state, keys, values)
            scores = model.get_word_scores(decoded, new_context)
            probs = -nn.LogSoftmax(dim=1)(scores).squeeze()
            probs = probs.detach().cpu().numpy()
            max_indices = np.argpartition(probs, beam_size - 1)[:beam_size]
            next_hypothesis = []
            for i in max_indices:
                if i < 0:
                    prob_idx = 2
                else:
                    prob_idx = i

                next_hypothesis.append(
                    (
                        current_sentence + [torch.LongTensor([i]).to(scores.device)],
                        current_state,
                        new_context,
                        score_update(current_score, probs[prob_idx], time_step)
                    )
                )
            next_hypotheses.extend(next_hypothesis)
        if stopped:
            break

        beam_scores = np.array([hypothesis[3] for hypothesis in next_hypotheses])
        beam_indices = np.argpartition(beam_scores, beam_size - 1)[:beam_size]
        hypotheses = [next_hypotheses[j] for j in beam_indices]

    return sorted([(torch.cat(hypothesis[0]), hypothesis[3]) for hypothesis in hypotheses], key=lambda x: x[1])


def score_update(old_score, update, time_step):

    return old_score * (time_step - 1) / time_step + update / time_step
