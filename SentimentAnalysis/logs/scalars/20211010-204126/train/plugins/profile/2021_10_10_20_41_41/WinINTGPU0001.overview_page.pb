?	0???]x@0???]x@!0???]x@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'0???]x@?:????@1?Z|
 x@I9?)9'v??r0*	?????[@2_
(Iterator::Root::MapAndBatch::TensorSlice???9#J??!3????H@)???9#J??13????H@:Preprocessing2R
Iterator::Root::MapAndBatchsh??|???!Мw^$C@)sh??|???1Мw^$C@:Preprocessing2E
Iterator::RootK?46??!?ip?EjI@)_?Qڋ?1?3???)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?]=?7??Q??S ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:????@?:????@!?:????@      ??!       "	?Z|
 x@?Z|
 x@!?Z|
 x@*      ??!       2      ??!       :	9?)9'v??9?)9'v??!9?)9'v??B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?]=?7??y??S ?X@?"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul?? ???!?? ???0"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMulߍO:??!?T?????0"Y
;model/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMul!m????!?r??R??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMuluZ[^??!CI,???0"X
:model/bert/encoder/layer_._5/output/dense/Tensordot/MatMulMatMulIQ?B???!?m?????0"X
:model/bert/encoder/layer_._6/output/dense/Tensordot/MatMulMatMulC??????!?`r?O??0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMul'ē*???!x??W????0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMul???_H??!?ͩch??0"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMulq?0???!ڮ?i?t??0"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMulJ??m?	??!"??????0Q      Y@Y????T@aq??5??W@q=(H???X@"?

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
Refer to the TF2 Profiler FAQb?99.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 