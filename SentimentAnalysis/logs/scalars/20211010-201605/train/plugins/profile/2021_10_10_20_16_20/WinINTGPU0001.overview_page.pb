?	F?n?qOx@F?n?qOx@!F?n?qOx@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'F?n?qOx@P6?
?@1??	?Hx@I'L5???r0*	      \@2_
(Iterator::Root::MapAndBatch::TensorSlice_?Qګ?!؂-؂-H@)_?Qګ?1؂-؂-H@:Preprocessing2R
Iterator::Root::MapAndBatch?e??a???!P??O??C@)?e??a???1P??O??C@:Preprocessing2E
Iterator::Root:??H???!(}?'}?I@)??Pk?w??1a?`?(@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?m???k??Q%?6?(?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	P6?
?@P6?
?@!P6?
?@      ??!       "	??	?Hx@??	?Hx@!??	?Hx@*      ??!       2      ??!       :	'L5???'L5???!'L5???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?m???k??y%?6?(?X@?"Y
;model/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMulkO?Y+??!kO?Y+??0"X
:model/bert/encoder/layer_._3/output/dense/Tensordot/MatMulMatMul??7*??!C????*??0"X
:model/bert/encoder/layer_._6/output/dense/Tensordot/MatMulMatMul٩w? ??!??憣]??0"X
:model/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMul:??^ ??!?d??%??0"X
:model/bert/encoder/layer_._0/output/dense/Tensordot/MatMulMatMul?,?(k??!?ft????0"X
:model/bert/encoder/layer_._1/output/dense/Tensordot/MatMulMatMulˎ0???!?)?h?Y??0"X
:model/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMul?g?<??!?V*P強?0"X
:model/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMulƨr9???!??X׷??0"Y
;model/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMulM?<??
??!kE?,???0"X
:model/bert/encoder/layer_._9/output/dense/Tensordot/MatMulMatMul?O?	i??!)?/???0Q      Y@Y????T@aq??5??W@q<??4??X@"?

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
Refer to the TF2 Profiler FAQb?99.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 