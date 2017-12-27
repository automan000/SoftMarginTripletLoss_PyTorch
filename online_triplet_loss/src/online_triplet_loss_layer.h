void online_triplet_loss_forward(
    THFloatTensor *bottom_data_tensor, // NxM features
    THIntTensor *label_tensor,
    THFloatTensor * top_tensor,
    
    // saved for backward
    THFloatTensor *dist_,   // (num, num)
    THIntTensor * triplets_tensor,  // ((num/2)^3+1, 3)
    THIntTensor * pos_pairs_tensor, // (num^2 + 1, 2)

    const int all_triplets,
    const int positive_type,
    const int negative_type,
    const int margin_type,
    const float margin,
    const float mu);

void online_triplet_loss_backward(
    THFloatTensor *top_grad_tensor,
    THFloatTensor *bottom_data_tensor,
    THFloatTensor *bottom_grad_tensor,
    THFloatTensor *agg_tensor, // (num, num)
    // saved for backward
    THFloatTensor *dist_,          // (num, num)
    THIntTensor *triplets_tensor,  // ((num/2)^3+1, 3)
    THIntTensor *pos_pairs_tensor, // (num^2 + 1, 2)
    
    const float mu);