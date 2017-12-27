#include <TH/TH.h>
#include <TH/THMath.h>
#include <TH/THBlas.h>
#include <stdio.h>
#include <math.h>
#include "online_triplet_loss_layer.h"
#include "online_triplet_loss.h"

void init2dTensor(THFloatTensor *tensor, const int height, const int width)
{
    if (tensor->nDimension != 2 || tensor->size[0] * tensor->size[1] < height * width)
    {
        THFloatTensor_resize2d(tensor, height, width);
    }
    THFloatTensor_zero(tensor);
}

void init2dIntTensor(THIntTensor *tensor, const int height, const int width)
{
    if (tensor->nDimension != 2 || tensor->size[0] * tensor->size[1] < height * width)
    {
        THIntTensor_resize2d(tensor, height, width);
    }
    THIntTensor_zero(tensor);
}

void online_triplet_loss_forward(
    THFloatTensor *bottom_data_tensor, // NxM features
    THIntTensor *label_tensor,
    THFloatTensor *top_tensor,

    // saved for backward
    THFloatTensor *dist_,          // (num, num)
    THIntTensor *triplets_tensor,  // ((num^3/4)+1, 3)
    THIntTensor *pos_pairs_tensor, // (num^2 + 1, 2)

    const int all_triplets,
    const int positive_type,
    const int negative_type,
    const int margin_type,
    const float margin,
    const float mu)
{
    const int num = bottom_data_tensor->size[0];
    const int dim = THFloatTensor_numel(bottom_data_tensor) / num;
    // const int dim = bottom_data_tensor->numel() / num;
    const int channels = bottom_data_tensor->size[1];

    // Init Output Size
    THFloatTensor_resize1d(top_tensor, 2);

    // Init triplets and pos_pairs
    // THFloatTensor_resize2d(dist_, num, num);
    // THIntTensor_resize2d(triplets_tensor, num * num * num / 4 + 2, 3);
    // THIntTensor_resize2d(pos_pairs_tensor, num * num + 1, 2);
    init2dTensor(dist_, num, num);
    init2dIntTensor(triplets_tensor, num * num * num / 4 + 2, 3);
    init2dIntTensor(pos_pairs_tensor, num * num + 1, 2);

    // Computing the pairwise Euclidean distance
    THFloatTensor *temp_diff = THFloatTensor_new();
    THFloatTensor_resize2d(temp_diff, 1, channels);

    THFloatTensor *feature_i = THFloatTensor_new();
    THFloatTensor *feature_j = THFloatTensor_new();
    int i, j;
    for (i = 0; i < num; i++)
    {
        float *dist_row_i = THFloatTensor_data(dist_) + i * num;

        // THFloatTensor *feature_i = THFloatTensor_newSelect(bottom_data_tensor, 0, i);
        THFloatTensor_select(feature_i, bottom_data_tensor, 0, i);
        for (j = 0; j < num; j++)
        {
            // THFloatTensor *feature_j = THFloatTensor_newSelect(bottom_data_tensor, 0, j);
            THFloatTensor_select(feature_j, bottom_data_tensor, 0, j);
            THFloatTensor_csub(temp_diff, feature_i, 1, feature_j);
            dist_row_i[j] = THFloatTensor_dot(temp_diff, temp_diff);
            // dist_row_i[j] = sqrt(dist_row_i[j]);
        }
    }

    // Loss
    OnlineTripletLossLoss_Forward(
        THFloatTensor_data(dist_),
        THIntTensor_data(label_tensor),
        THFloatTensor_data(top_tensor),

        THIntTensor_data(triplets_tensor),
        THIntTensor_data(pos_pairs_tensor),

        num,
        dim,
        channels,

        all_triplets,
        positive_type,
        negative_type,
        margin_type,
        margin,
        mu);

    THFloatTensor_free(temp_diff);
    THFloatTensor_free(feature_i);
    THFloatTensor_free(feature_j);
}

void online_triplet_loss_backward(
    THFloatTensor *top_grad_tensor,
    THFloatTensor *bottom_data_tensor,
    THFloatTensor *bottom_grad_tensor,
    THFloatTensor *agg_tensor, // (num, num)
    // saved for backward
    THFloatTensor *dist_,          // (num, num)
    THIntTensor *triplets_tensor,  // ((num/2)^3+1, 3)
    THIntTensor *pos_pairs_tensor, // (num^2 + 1, 2)

    const float mu)
{
    // num_triplets_, triplets_size, pos_pairs_size
    int *triplets_data = THIntTensor_data(triplets_tensor);
    int *pos_pairs_data = THIntTensor_data(pos_pairs_tensor);
    const int num_triplets_ = triplets_data[1];
    const int triplets_size = triplets_data[0];
    const int pos_pairs_size = pos_pairs_data[0];

    // bottom size
    float *bottom_data = THFloatTensor_data(bottom_data_tensor);
    const int num = bottom_data_tensor->size[0];
    const int dim = THFloatTensor_numel(bottom_data_tensor) / num;

    // init output size
    init2dTensor(bottom_grad_tensor, num, dim);

    init2dTensor(agg_tensor, num, num);
    float *agg_data = THFloatTensor_data(agg_tensor);

    if (num_triplets_ > 0)
    {
        float scale1 = 2.0 / num_triplets_ * mu;
        for (int i = 0; i < triplets_size; ++i)
        {
            // int qry_id = triplets_[i].first_;
            // int pos_id = triplets_[i].second_;
            // int neg_id = triplets_[i].third_;
            int *row = triplets_data + 3 * (i + 1);
            int qry_id = row[0];
            int pos_id = row[1];
            int neg_id = row[2];

            agg_data[qry_id * num + neg_id] += scale1;
            agg_data[qry_id * num + pos_id] -= scale1;

            agg_data[pos_id * num + pos_id] += scale1;
            agg_data[pos_id * num + qry_id] -= scale1;

            agg_data[neg_id * num + qry_id] += scale1;
            agg_data[neg_id * num + neg_id] -= scale1;
        }
    }

    if (pos_pairs_size > 0)
    {
        float scale2 = 2.0 / pos_pairs_size * (1.0 - mu);
        for (int i = 0; i < pos_pairs_size; ++i)
        {
            //   int qry_id = pos_pairs_[i].first;
            //   int pos_id = pos_pairs_[i].second;
            int *row = pos_pairs_data + 2 * (i + 1);
            int qry_id = row[0];
            int pos_id = row[1];

            agg_data[qry_id * num + qry_id] += scale2;
            agg_data[qry_id * num + pos_id] -= scale2;

            agg_data[pos_id * num + pos_id] += scale2;
            agg_data[pos_id * num + qry_id] -= scale2;
        }
    }

    const float loss_weight = THFloatTensor_data(top_grad_tensor)[0];

    // row major: [num, num] x [num, dim] = [num, dim]
    // column major: [dim, num] x [num, num] = [dim, num]
    THFloatBlas_gemm('n', 'n', dim, num, num, loss_weight,
                bottom_data, dim,
                agg_data, num,
                0,
                THFloatTensor_data(bottom_grad_tensor), dim
            );
}
