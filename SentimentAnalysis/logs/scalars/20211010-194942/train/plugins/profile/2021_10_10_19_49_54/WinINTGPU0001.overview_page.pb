?	zm6V?Cx@zm6V?Cx@!zm6V?Cx@		??_?ґ?	??_?ґ?!	??_?ґ?"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0zm6V?Cx@?~???s@1K?8??x@I*?D/?X??YJ??%?L??r0*	     ?a@2R
Iterator::Root::MapAndBatch]m???{??!?$I?$?I@)]m???{??1?$I?$?I@:Preprocessing2_
(Iterator::Root::MapAndBatch::TensorSlice??6???!?m۶m[C@)??6???1?m۶m[C@:Preprocessing2E
Iterator::RootI.?!????!I?$I??N@)_?Qڋ?1۶m۶m#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9
??_?ґ?I 7?????Qn/??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~???s@?~???s@!?~???s@      ??!       "	K?8??x@K?8??x@!K?8??x@*      ??!       2      ??!       :	*?D/?X??*?D/?X??!*?D/?X??B      ??!       J	J??%?L??J??%?L??!J??%?L??R      ??!       Z	J??%?L??J??%?L??!J??%?L??b      ??!       JGPUY
??_?ґ?b q 7?????yn/??X@?"X
:model/bert/encoder/layer_._6/output/dense/Tensordot/MatMulMatMul+?ݙ?5??!+?ݙ?5??0"X
:model/bert/encoder/layer_._4/output/dense/Tensordot/MatMulMatMul?}?0?4??!??Xe>5??0"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMulw?h?(??!????d??0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul?????$??!Y???-??0"X
:model/bert/encoder/layer_._5/output/dense/Tensordot/MatMulMatMul?????#??!? ??Y???0"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMul???H?"??!7v
??_??0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMul3??? ??!ݵ???÷?0"Y
;model/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMulalNe??!i?1.?&??0"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMul? Xq??!	?1Y?4??0"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul????Vl??!]F@???0Q      Y@Y????T@aq??5??W@q??^0^?X@"?

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
Refer to the TF2 Profiler FAQb?98.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 