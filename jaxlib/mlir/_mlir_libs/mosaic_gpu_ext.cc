/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdint>
#include <vector>

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"  // IWYU pragma: keep
#include "mlir/IR/Block.h"  // IWYU pragma: keep
#include "mlir/IR/Location.h"  // IWYU pragma: keep
#include "mlir/IR/Operation.h"  // IWYU pragma: keep
#include "mlir/IR/Value.h"  // IWYU pragma: keep
#include "nanobind/nanobind.h"
#include "nanobind/operators.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/tuple.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "jaxlib/mosaic/dialect/gpu/integrations/c/attributes.h"
#include "jaxlib/mosaic/dialect/gpu/integrations/c/gpu_dialect.h"


namespace nb = nanobind;


NB_MODULE(_mosaic_gpu_ext, m) {
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle dialect = mlirGetDialectHandle__mosaic_gpu__();
        mlirDialectHandleRegisterDialect(dialect, context);
        if (load) {
          mlirDialectHandleLoadDialect(dialect, context);
        }
      },
      nb::arg("context"), nb::arg("load") = true);

  m.def("register_inliner_extensions", [](MlirContext context) {
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    mlirDialectRegistryInsertMosaicGpuInlinerExtensions(registry);
    mlirContextAppendDialectRegistry(context, registry);
    mlirDialectRegistryDestroy(registry);
  });

  auto barrier_type =
      mlir::python::nanobind_adaptors::mlir_type_subclass(
          m, "BarrierType", mlirMosaicGpuIsABarrierType,
          mlirMosaicGpuBarrierTypeGetTypeID);
  barrier_type
      .def_staticmethod(
          "get",
          [cls = barrier_type.get_class()](bool orders_tensor_core,
                                           MlirContext ctx) {
            return cls(mlirMosaicGpuBarrierTypeGet(ctx, orders_tensor_core));
          },
          nb::arg("orders_tensor_core") = false,
          nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "orders_tensor_core: bool = False, "
              "context: mlir.ir.Context | None = None"
              ") -> BarrierType"
              // clang-format: on
              ),
          "Creates a BarrierType.")
      .def_property_readonly("orders_tensor_core",
                             mlirMosaicGpuBarrierTypeGetOrdersTensorCore);

  auto tile_transform_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "TileTransformAttr", mlirMosaicGpuIsATileTransformAttr,
          mlirMosaicGpuTileTransformAttrGetTypeID);
  tile_transform_attr
      .def_staticmethod(
          "get",
          [cls = tile_transform_attr.get_class()](std::vector<int32_t>& tiling,
                                                  MlirContext ctx) {
            return cls(mlirMosaicGpuTileTransformAttrGet(ctx, tiling.data(),
                                                         tiling.size()));
          },
          nb::arg("tiling"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "tiling: Sequence[int], "
              "context: mlir.ir.Context | None = None"
              ") -> TileTransformAttr"
              // clang-format: on
              ),
          "Creates a TileTransformAttr with the given tiling.")
      .def_property_readonly(
          "tiling", mlirMosaicGpuTileTransformAttrGetTiling,
          nb::sig("def tiling(self) -> mlir.ir.DenseI32ArrayAttr"));

  auto transpose_transform_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "TransposeTransformAttr", mlirMosaicGpuIsATransposeTransformAttr,
          mlirMosaicGpuTransposeTransformAttrGetTypeID);
  transpose_transform_attr
      .def_staticmethod(
          "get",
          [cls = transpose_transform_attr.get_class()](
              std::vector<int32_t>& permutation, MlirContext ctx) {
            return cls(mlirMosaicGpuTransposeTransformAttrGet(
                ctx, permutation.data(), permutation.size()));
          },
          nb::arg("permutation"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "permutation: Sequence[int], "
              "context: mlir.ir.Context | None = None"
              ") -> TransposeTransformAttr"
              // clang-format: on
              ),
          "Creates a TransposeTransformAttr with the given permutation.")
      .def_property_readonly(
          "permutation", mlirMosaicGpuTransposeTransformAttrGetPermutation,
          nb::sig("def permutation(self) -> mlir.ir.DenseI32ArrayAttr"));

  auto swizzle_transform_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "SwizzleTransformAttr", mlirMosaicGpuIsASwizzleTransformAttr,
          mlirMosaicGpuSwizzleTransformAttrGetTypeID);
  swizzle_transform_attr
      .def_staticmethod(
          "get",
          [cls = swizzle_transform_attr.get_class()](int32_t swizzle,
                                                     MlirContext ctx) {
            return cls(mlirMosaicGpuSwizzleTransformAttrGet(ctx, swizzle));
          },
          nb::arg("swizzle"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "swizzle: int, "
              "context: mlir.ir.Context | None = None"
              ") -> SwizzleTransformAttr"
              // clang-format: on
              ),
          "Creates a SwizzleTransformAttr with the given swizzle.")
      .def_property_readonly("swizzle",
                             mlirMosaicGpuSwizzleTransformAttrGetSwizzle);

  auto splat_fragmented_layout_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "WGSplatFragLayoutAttr", mlirMosaicGpuIsAWGSplatFragLayoutAttr,
          mlirMosaicGpuWGSplatFragLayoutAttrGetTypeID);
  splat_fragmented_layout_attr
      .def_staticmethod(
          "get",
          [cls = splat_fragmented_layout_attr.get_class()](MlirAttribute shape,
                                                           MlirContext ctx) {
            return cls(mlirMosaicGpuWGSplatFragLayoutAttrGet(ctx, shape));
          },
          nb::arg("shape"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "shape: mlir.ir.DenseI64ArrayAttr, "
              "context: mlir.ir.Context | None = None"
              ") -> WGSplatFragLayoutAttr"
              // clang-format: on
              ),
          "Creates a WGSplatFragLayoutAttr with the given shape.")
      .def_property_readonly(
          "shape", mlirMosaicGpuWGSplatFragLayoutAttrGetShape,
          nb::sig("def shape(self) -> mlir.ir.DenseI64ArrayAttr"));

  auto strided_fragmented_layout_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "WGStridedFragLayoutAttr", mlirMosaicGpuIsAWGStridedFragLayoutAttr,
          mlirMosaicGpuWGStridedFragLayoutAttrGetTypeID);
  strided_fragmented_layout_attr
      .def_staticmethod(
          "get",
          [cls = strided_fragmented_layout_attr.get_class()](
              MlirAttribute shape, int32_t vector_size, MlirContext ctx) {
            return cls(mlirMosaicGpuWGStridedFragLayoutAttrGet(ctx, shape,
                                                               vector_size));
          },
          nb::arg("shape"), nb::arg("vector_size"),
          nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "shape: mlir.ir.DenseI64ArrayAttr, "
              "vector_size: int, "
              "context: mlir.ir.Context | None = None"
              ") -> WGStridedFragLayoutAttr"
              // clang-format: on
              ),
          "Creates a WGStridedFragLayoutAttr.")
      .def_property_readonly(
          "shape", mlirMosaicGpuWGStridedFragLayoutAttrGetShape,
          nb::sig("def shape(self) -> mlir.ir.DenseI64ArrayAttr"))
      .def_property_readonly("vector_size",
                             mlirMosaicGpuWGStridedFragLayoutAttrGetVectorSize);

  auto replicated_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "ReplicatedAttr", mlirMosaicGpuIsAReplicatedAttr,
          mlirMosaicGpuReplicatedAttrGetTypeID);
  replicated_attr
      .def_staticmethod(
          "get",
          [cls = replicated_attr.get_class()](int32_t times, MlirContext ctx) {
            return cls(mlirMosaicGpuReplicatedAttrGet(ctx, times));
          },
          nb::arg("times"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "times: int, "
              "context: mlir.ir.Context | None = None"
              ") -> ReplicatedAttr"
              // clang-format: on
              ),
          "Creates a ReplicatedAttr.")
      .def_property_readonly("times", mlirMosaicGpuReplicatedAttrGetTimes);

  auto tiled_layout_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "TiledLayoutAttr", mlirMosaicGpuIsATiledLayoutAttr,
          mlirMosaicGpuTiledLayoutAttrGetTypeID);
  tiled_layout_attr
      .def_staticmethod(
          "get",
          [cls = tiled_layout_attr.get_class()](
              MlirAttribute tiling, MlirAttribute warp_dims,
              MlirAttribute lane_dims, int32_t vector_dim, MlirContext ctx) {
            return cls(mlirMosaicGpuTiledLayoutAttrGet(ctx, tiling, warp_dims,
                                                       lane_dims, vector_dim));
          },
          nb::arg("tiling"), nb::arg("warp_dims"), nb::arg("lane_dims"),
          nb::arg("vector_dim"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "tiling: mlir.ir.ArrayAttr, "
              "warp_dims: mlir.ir.ArrayAttr, "
              "lane_dims: mlir.ir.ArrayAttr, "
              "vector_dim: int, "
              "context: mlir.ir.Context | None = None"
              ") -> TiledLayoutAttr"
              // clang-format: on
              ),
          "Creates a TiledLayoutAttr.")
      .def_property_readonly("tiling", mlirMosaicGpuTiledLayoutAttrGetTiling,
                             nb::sig("def tiling(self) -> mlir.ir.ArrayAttr"))
      .def_property_readonly(
          "warp_dims", mlirMosaicGpuTiledLayoutAttrGetWarpDims,
          nb::sig("def warp_dims(self) -> mlir.ir.ArrayAttr"))
      .def_property_readonly(
          "lane_dims", mlirMosaicGpuTiledLayoutAttrGetLaneDims,
          nb::sig("def lane_dims(self) -> mlir.ir.ArrayAttr"))
      .def_property_readonly("vector_dim",
                             mlirMosaicGpuTiledLayoutAttrGetVectorDim);

  auto copy_partition_attr_interface =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "CopyPartitionAttrInterface", mlirMosaicGpuIsACopyPartitionAttr);

  auto copy_replicated_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "CopyReplicatedAttr", mlirMosaicGpuIsACopyReplicatedAttr,
          copy_partition_attr_interface.get_class(),
          mlirMosaicGpuCopyReplicatedAttrGetTypeID);
  copy_replicated_attr.def_staticmethod(
      "get",
      [cls = copy_replicated_attr.get_class()](MlirContext ctx) {
        return cls(mlirMosaicGpuCopyReplicatedAttrGet(ctx));
      },
      nb::arg("context").none() = nb::none(),
      nb::sig(
          // clang-format: off
          "def get("
          "context: mlir.ir.Context | None = None"
          ") -> CopyReplicatedAttr"
          // clang-format: on
          ),
      "Creates a CopyReplicatedAttr.");

  auto copy_partitioned_attr =
      mlir::python::nanobind_adaptors::mlir_attribute_subclass(
          m, "CopyPartitionedAttr", mlirMosaicGpuIsACopyPartitionedAttr,
          copy_partition_attr_interface.get_class(),
          mlirMosaicGpuCopyPartitionedAttrGetTypeID);
  copy_partitioned_attr
      .def_staticmethod(
          "get",
          [cls = copy_partitioned_attr.get_class()](int32_t axis,
                                                    MlirContext ctx) {
            return cls(mlirMosaicGpuCopyPartitionedAttrGet(ctx, axis));
          },
          nb::arg("axis"), nb::arg("context").none() = nb::none(),
          nb::sig(
              // clang-format: off
              "def get("
              "axis: int, "
              "context: mlir.ir.Context | None = None"
              ") -> CopyPartitionedAttr"
              // clang-format: on
              ),
          "Creates a CopyPartitionedAttr.")
      .def_property_readonly("axis", mlirMosaicGpuCopyPartitionedAttrGetAxis);
}
