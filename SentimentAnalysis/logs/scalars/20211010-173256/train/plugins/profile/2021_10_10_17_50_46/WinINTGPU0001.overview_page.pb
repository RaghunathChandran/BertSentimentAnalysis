?	????s]x@????s]x@!????s]x@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'????s]x@???^D?@1?w???!x@I???????r0*	?????YY@2_
(Iterator::Root::MapAndBatch::TensorSlice??e?c]??!?7<KQK@)??e?c]??1?7<KQK@:Preprocessing2R
Iterator::Root::MapAndBatch????Mb??!۽=???@)????Mb??1۽=???@:Preprocessing2E
Iterator::RootZd;?O???!u??ô?F@)y?&1???1 ?u??+@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?1?KT???Q?MhW??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???^D?@???^D?@!???^D?@      ??!       "	?w???!x@?w???!x@!?w???!x@*      ??!       2      ??!       :	??????????????!???????B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?1?KT???y?MhW??X@?"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMul?d?????!?d?????0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMul@?????!aҬ]N??0"X
:model/bert/encoder/layer_._0/output/dense/Tensordot/MatMulMatMul2???"??!???oR??0"X
:model/bert/encoder/layer_._5/output/dense/Tensordot/MatMulMatMul?-U???!?X????0"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMul???????!R??2i???0"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMulf^uć??!?+zO??0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMul???[P??!??6d???0"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMul>:g????!??PM??0"X
:model/bert/encoder/layer_._6/output/dense/Tensordot/MatMulMatMul}??eg??!.S:?t??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul??嬪
??!???????0Q      Y@Y????T@aq??5??W@qn	????X@"?

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