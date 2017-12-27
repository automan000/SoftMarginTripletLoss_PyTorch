#ifndef _ONLINE_TRIPLET_LOSS
#define _ONLINE_TRIPLET_LOSS

 #ifdef __cplusplus
 extern "C" {
 #endif

#define Dtype float

void OnlineTripletLossLoss_Forward(
    Dtype *dist_, // (num, num)
    int *label, // (num,)
    Dtype *loss_data,   // (2,) loss, accuracy
    int * triplets_,    // ((num/2)^3+1, 3)
    int * pos_pairs_,   // (num^2 + 1, 2)
    const int num,      // size(0)
    const int dim,      // (size(0,1,2,3)) / size(0)
    const int channels, // size(1)
    const int all_triplets,
    const int positive_type,
    const int negative_type,
    const int margin_type,
    const float margin,
    const float mu);

 #ifdef __cplusplus
 }
 #endif

#endif