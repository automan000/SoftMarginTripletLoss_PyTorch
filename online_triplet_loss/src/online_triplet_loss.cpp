#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include <math.h>

#include "online_triplet_loss.h"

using namespace std;

enum TripletLossParameter_SampleMethod
{
    TripletLossParameter_SampleMethod_ALL = 0,
    TripletLossParameter_SampleMethod_HARD = 1,
    TripletLossParameter_SampleMethod_MODERATE = 2
};

enum TripletLossParameter_MarginType
{
    TripletLossParameter_MarginType_HARD = 0,
    TripletLossParameter_MarginType_SOFTMARGIN = 1
};


class Triplet {
public:
  explicit Triplet(int first, int second, int third) :
    first_(first), second_(second), third_(third) {
  }
  int first_;
  int second_;
  int third_;
};

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
    const float mu)
{

    // Dtype num_triplets_;
    // vector<Triplet> triplets_;
    // vector<pair<int, int> > pos_pairs_;

    /**
    * Find boundary of different classes.
    * A batch is composed by small groups with items belongs to the same class.
    *  e.g. gruop size is 3, batch size is 15:
    *    1 1 1 2 2 2 3 3 3 1 1 1 4 4 4
    */
    vector<int> boundary;
    int prev = Dtype(-1);
    for (int i = 0; i < num; ++i)
    {
        if (prev != label[i])
        {
            boundary.push_back(i);
            prev = label[i];
        }
    }
    boundary.push_back(num);

    //calculate mean distances within each class
    vector<Dtype> mean_distances;
    vector<pair<Dtype, Dtype>> min_max_distances;
    // classes
    for (int c = 0; c < boundary.size() - 1; c++)
    {
        // query
        Dtype sum_dist = 0.0;
        Dtype min_dist = 2.0;
        Dtype max_dist = 0.0;
        int pos_count = 0;
        for (int i = boundary[c]; i < boundary[c + 1]; ++i)
        {
            const Dtype *dist_data = dist_ + i * num;
            for (int j = boundary[c]; j < boundary[c + 1]; ++j)
            {
                if (i == j)
                    continue;
                if (dist_data[j] < min_dist)
                    min_dist = dist_data[j];
                if (dist_data[j] > max_dist)
                    max_dist = dist_data[j];
                sum_dist += dist_data[j];
                pos_count++;
            }
        }
        mean_distances.push_back(sum_dist / static_cast<Dtype>(pos_count));
        min_max_distances.push_back(pair<Dtype, Dtype>(min_dist, max_dist));
    }

    /**
    * Sampling pairs and triplets, then computing the loss
    */
    Dtype pair_loss = Dtype(0);
    Dtype rank_loss = Dtype(0);
    Dtype cur_rank_loss = Dtype(0);
    Dtype pos_dist = Dtype(0);
    Dtype neg_dist = Dtype(0);
    Dtype one_minus_mu = Dtype(1) - mu;
    //pairwise loss
    // pos_pairs_.clear();
    int pos_pairs_size = 0;
    if (one_minus_mu > Dtype(0))
    {
        // classes
        for (int c = 0; c < boundary.size() - 1; c++)
        {
            // query
            for (int i = boundary[c]; i < boundary[c + 1]; ++i)
            {
                const Dtype *dist_data = dist_ + i * num;
                // positive
                for (int j = boundary[c]; j < boundary[c + 1]; ++j)
                {
                    if (i == j)
                    {
                        continue;
                    }
                    pair_loss += dist_data[j];
                    // pos_pairs_.push_back(pair<int, int>(i, j));
                    int idx = (pos_pairs_size+1) * 2;
                    pos_pairs_[idx + 0] = i;
                    pos_pairs_[idx + 1] = j;
                    pos_pairs_size++;
                }
            }
        }
    }
    pos_pairs_[0] = pos_pairs_size;
    // pair_loss = pos_pairs_.size() > 0 ? pair_loss / pos_pairs_.size() : 0;
    pair_loss = pos_pairs_size > 0 ? pair_loss / pos_pairs_size : 0;

    //triplet loss
    // triplets_.clear();
    int triplets_size = 0;
    int all_triplet_size = 0;
    int num_error = 0;
    // classes
    for (int c = 0; c < boundary.size() - 1; c++)
    {
        // query
        Dtype hard_pos_threshold = 2 * mean_distances[c] - min_max_distances[c].first;
        for (int i = boundary[c]; i < boundary[c + 1]; ++i)
        {
            const Dtype *dist_data = dist_ + i * num;
            // positive
            for (int j = boundary[c]; j < boundary[c + 1]; ++j)
            {
                if (i == j)
                {
                    continue;
                }
                pos_dist = dist_data[j];
                switch (positive_type)
                {
                case TripletLossParameter_SampleMethod_ALL:
                    break;
                case TripletLossParameter_SampleMethod_HARD:
                    //sample the positives whose distance greater than the average
                    if (pos_dist < mean_distances[c])
                        continue;
                    break;
                case TripletLossParameter_SampleMethod_MODERATE:
                    //sample the positives without the hardest ones
                    if (pos_dist > hard_pos_threshold)
                        continue;
                    break;
                default:
                    cout << "Unknown positive sampling method: " << positive_type;
                    exit(-1);
                }

                // negative groups
                for (int m = 0; m < boundary.size() - 1; m++)
                {
                    if (label[boundary[m]] == label[i])
                    {
                        continue;
                    }
                    // negative
                    for (int k = boundary[m]; k < boundary[m + 1]; ++k)
                    {
                        all_triplet_size++;
                        neg_dist = dist_data[k];
                        cur_rank_loss = margin + pos_dist - neg_dist;
                        num_error += (pos_dist >= neg_dist);
                        // cout<<"test"<<endl;
                        if (margin_type == TripletLossParameter_MarginType_HARD)
                        	cout<<"hard:"<<endl;
                        	if(cur_rank_loss < 0)
                            	continue; //not violate the loss
                        else
                        {
                        	//cout<<"soft mode:"<<endl;
                        	//cout<<"raw loss:"<<cur_rank_loss<<endl;
                        	//cout<<"exp:"<<exp(cur_rank_loss)<<endl;
                        	//cout<<"1+exp:"<<1+exp(cur_rank_loss)<<endl;
                        	cur_rank_loss = log(1+exp(cur_rank_loss));
                        	//cout<<"single loss:"<<cur_rank_loss<<endl;
                        }

                        switch (negative_type)
                        {
                        case TripletLossParameter_SampleMethod_ALL:
                            break;
                        case TripletLossParameter_SampleMethod_HARD:
                            //sample the negatives whose distance smaller than the positive one
                            if (neg_dist > pos_dist)
                                continue;
                            break;
                        case TripletLossParameter_SampleMethod_MODERATE:
                            //sample the positives without the hardest ones
                            if (neg_dist <= pos_dist)
                                continue;
                            break;
                        default:
                            cout << "Unknown negative sampling method.";
                            exit(-1);
                        }
                        rank_loss += cur_rank_loss;
                        // triplets_.push_back(Triplet(i, j, k));
                        int idx = (triplets_size+1) * 3;
                        triplets_[idx + 0] = i;
                        triplets_[idx + 1] = j;
                        triplets_[idx + 2] = k;
                        triplets_size++;

                    } // end of negative
                }     // end of negative groups
            }         // end of positive
        }             // end of query
    }                 // end of classes
    triplets_[0] = triplets_size;

    int num_triplets_;
    if (all_triplets)
        // num_triplets_ = static_cast<Dtype>(all_triplet_size); //triplets_.size(); //
        num_triplets_ = all_triplet_size;
    else
        // num_triplets_ = static_cast<Dtype>(triplets_.size());
        num_triplets_ = triplets_size;
    triplets_[1] = num_triplets_;

    rank_loss = num_triplets_ > 0 ? rank_loss / num_triplets_ : 0;
	// cout<<"batch loss:"<<rank_loss<<endl;


    // average loss among all triplets
    loss_data[0] = rank_loss * mu + pair_loss * one_minus_mu;
    // average accuracy among all triplets
    loss_data[1] = Dtype(1) - (all_triplet_size > 0 ? Dtype(num_error) / all_triplet_size : 0);

    // TODO: Save
    // Dtype num_triplets_;
    // vector<Triplet> triplets_;
    // vector<pair<int, int> > pos_pairs_;
}
