?	???46x@???46x@!???46x@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???46x@;9Cq??@1?<Y??w@Ini5$????r0*??????Y@)      =2_
(Iterator::Root::MapAndBatch::TensorSliceݵ?|г??!?%mI[RH@)ݵ?|г??1?%mI[RH@:Preprocessing2R
Iterator::Root::MapAndBatch8??d?`??!???!oHC@)8??d?`??1???!oHC@:Preprocessing2E
Iterator::Root?A`??"??!Lڒ???I@)F%u???1fMYS֔)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI sí???Q4?H??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;9Cq??@;9Cq??@!;9Cq??@      ??!       "	?<Y??w@?<Y??w@!?<Y??w@*      ??!       2      ??!       :	ni5$????ni5$????!ni5$????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q sí???y4?H??X@?"X
:model/bert/encoder/layer_._0/output/dense/Tensordot/MatMulMatMul?"3?DD??!?"3?DD??0"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMul ?(y????!????A??0"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMul?:ᙙ:??!KO??o??0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMul?J??7??!??! ?=??0"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul,??g?*??!??? ??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul?H?6e???!??C???0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMulWTe?n???!i??[%??0"Y
;model/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMulXup??!ʎ??5??0"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMulXup??!??|Z?E??0"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMul?:@ ~??!B1?*??0Q      Y@Y????T@aq??5??W@q???4X@"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 