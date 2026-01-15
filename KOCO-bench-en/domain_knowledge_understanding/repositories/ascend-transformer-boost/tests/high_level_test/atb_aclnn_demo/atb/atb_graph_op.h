#ifndef ATB_GRAPH_OP_H
#define ATB_GRAPH_OP_H
#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"
// 在构造图参数时，有两个点需要重点关注。一是Tensor的ID，ATB图接口中把Tensor分为三种类型，输入、输出和中间tensor，顾名思义，输入输出tensor是整图的输入输出tensor，
// 中间tensor则是在整图内的Tensor。构图时的TensorID从小到大应保证//为输入tensor、输出tensor、中间tensor的顺序，且每一种Tensor的个数要与参数中设置的一致。
// 二是要注意排布Node的顺序，用户需要根据计算图的拓扑结构把计算图变成一个有序队列，同时还要保证tensor与节点之间的关系和计算图保持一致。
atb::Status CreateGraphOperation(atb::Operation **operation);
#endif