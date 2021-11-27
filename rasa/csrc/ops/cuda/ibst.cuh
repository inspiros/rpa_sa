#pragma once

#include <ATen/ATen.h>

namespace rasa {
    template<typename scalar_t, typename index_t = int64_t>
    struct IBSTNode {
        index_t id;
        scalar_t data;
        std::shared_ptr <IBSTNode<scalar_t>> left = nullptr;
        std::shared_ptr <IBSTNode<scalar_t>> right = nullptr;

        IBSTNode(index_t id, scalar_t data, std::shared_ptr <IBSTNode<scalar_t>> left = nullptr,
                 std::shared_ptr <IBSTNode<scalar_t>> right = nullptr) {
            this->id = id;
            this->data = data;
            this->left = left;
            this->right = right;
        }

        inline bool is_leaf() {
            return this->left == nullptr && this->right == nullptr;
        }

        inline index_t left_id() {
            return this->left != nullptr ? this->left->id : -1;
        }

        inline index_t right_id() {
            return this->right != nullptr ? this->right->id : -1;
        }

        inline index_t get_index(const scalar_t data) const {
            return (data >= this->data)
                   ? (this->right != nullptr ? this->right->get_index(data) : this->id)
                   : (this->left != nullptr ? this->left->get_index(data) : this->id - 1);
        }
    };

    template<typename scalar_t, typename index_t = int64_t>
    inline std::shared_ptr <IBSTNode<scalar_t>> sorted_to_ibst(
            const at::TensorAccessor<scalar_t, 1> sorted,
            index_t start,
            index_t end) {
        if (start > end)
            return nullptr;
        index_t median_ind = (start + end) / 2;
        std::shared_ptr <IBSTNode<scalar_t>> root(new IBSTNode<scalar_t>(
                                                          median_ind,
                                                          sorted[median_ind],
                                                          sorted_to_ibst<scalar_t>(sorted, start, median_ind - 1),
                                                          sorted_to_ibst<scalar_t>(sorted, median_ind + 1, end)
                                                  )
        );
        return root;
    }

    template<typename scalar_t>
    static void _in_order_traverse(
            at::Tensor &trace,
            const std::shared_ptr <IBSTNode<scalar_t>> root = nullptr) {
        if (root == nullptr)
            return;
        _in_order_traverse(trace, root->left);
        trace[root->id][0] = root->data;
        trace[root->id][1] = root->left_id();
        trace[root->id][2] = root->right_id();
        _in_order_traverse(trace, root->right);
    }

    template<typename scalar_t, typename index_t = int64_t>
    class IBST {
        at::TensorOptions options;

    public:
        index_t n_nodes;
        std::shared_ptr <IBSTNode<scalar_t>> root = nullptr;

        IBST(const at::Tensor &sorted) {
            TORCH_CHECK(
                    sorted.ndimension() == 1,
                    "IBST: input tensor must be of 1 dimension; got",
                    sorted.ndimension());
            TORCH_CHECK(
                    sorted.is_cpu(),
                    "IBST: input tensor must be cpu"
            )
            this->n_nodes = sorted.numel();
            this->root = sorted_to_ibst<scalar_t>(sorted.accessor<scalar_t, 1>(), static_cast<index_t>(0),
                                                  this->n_nodes - 1);
            this->options = sorted.options();
        }

        inline index_t root_id() const {
            return this->root->id;
        }

        inline index_t get_index(const scalar_t data) const {
            return this->root->get_index(data);
        }

        at::Tensor in_order_traverse() {
            at::Tensor trace = at::empty({this->n_nodes, 3}, this->options);
            _in_order_traverse(trace, this->root);
            return trace;
        }
    };

    template<typename scalar_t, typename index_t = int64_t>
    __device__ __host__ index_t get_index_from_traversal_trace(
            at::GenericPackedTensorAccessor<scalar_t, 2> bst_traversal_trace,
            const index_t root_id,
            const scalar_t data) {
        index_t node_id = root_id;
        while (true) {
            auto node = bst_traversal_trace[node_id];
            if (data >= node[0]) {
                if (node[2] < 0)
                    break;
                node_id = static_cast<index_t>(node[2]);
            } else {
                if (node[1] < 0) {
                    node_id -= 1;
                    break;
                }
                node_id = static_cast<index_t>(node[1]);
            }
        }
        return node_id;
    }
}
