?	?ܚt??z@?ܚt??z@!?ܚt??z@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?ܚt??z@?4???+@1??)X#?y@I?st4 @r0*	?????LN@2_
(Iterator::Root::MapAndBatch::TensorSlicem???{???!?m?LkOK@)m???{???1?m?LkOK@:Preprocessing2R
Iterator::Root::MapAndBatch??~j?t??!?ngZ?@)??~j?t??1?ngZ?@:Preprocessing2E
Iterator::Root)\???(??!*????F@)?? ?rh??1?옥?,@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI ?;?6l@Qx!NJ?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?4???+@?4???+@!?4???+@      ??!       "	??)X#?y@??)X#?y@!??)X#?y@*      ??!       2      ??!       :	?st4 @?st4 @!?st4 @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?;?6l@yx!NJ?X@?"^
@model/bert/encoder/layer_._6/intermediate/dense/Tensordot/MatMulMatMul??/?!p~?!??/?!p~?0"_
Amodel/bert/encoder/layer_._10/intermediate/dense/Tensordot/MatMulMatMulp??Vh~?!][c<l??0"^
@model/bert/encoder/layer_._5/intermediate/dense/Tensordot/MatMulMatMulD?=XVh~?!???3Ж?0"_
Amodel/bert/encoder/layer_._11/intermediate/dense/Tensordot/MatMulMatMul????$c~?!?????h??0"^
@model/bert/encoder/layer_._8/intermediate/dense/Tensordot/MatMulMatMulWP?$c~?!???? ??0"^
@model/bert/encoder/layer_._9/intermediate/dense/Tensordot/MatMulMatMulWP?$c~?!?U?}Gͦ?0"^
@model/bert/encoder/layer_._7/intermediate/dense/Tensordot/MatMulMatMul|?Yx?X~?!b??l_???0"^
@model/bert/encoder/layer_._0/intermediate/dense/Tensordot/MatMulMatMul?"?M?C~?!??6?`??0"^
@model/bert/encoder/layer_._4/intermediate/dense/Tensordot/MatMulMatMulOt?(]A~?!?=?????0"^
@model/bert/encoder/layer_._2/intermediate/dense/Tensordot/MatMulMatMul??2?>~?! j?-q???0Q      Y@Y????T@aq??5??W@q?](\ETU@y??R?sLT?"?

both?Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?85.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 