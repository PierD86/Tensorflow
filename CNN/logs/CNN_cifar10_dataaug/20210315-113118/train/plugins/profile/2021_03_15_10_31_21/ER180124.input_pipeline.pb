	[Ӽ?Xy@[Ӽ?Xy@![Ӽ?Xy@	?2	?????2	????!?2	????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$[Ӽ?Xy@?v??/??A??+eOy@Y$???~???*	????)??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator^?I3O@!e???'?X@)^?I3O@1e???'?X@:Preprocessing2F
Iterator::Modelq?-???!???1???)??|гY??1A?????:Preprocessing2P
Iterator::Model::PrefetchZd;?O???!Í?
!ڢ?)Zd;?O???1Í?
!ڢ?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?c]?F3O@!??s?^?X@)?J?4a?1?~?VX?k?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?2	????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v??/???v??/??!?v??/??      ??!       "      ??!       *      ??!       2	??+eOy@??+eOy@!??+eOy@:      ??!       B      ??!       J	$???~???$???~???!$???~???R      ??!       Z	$???~???$???~???!$???~???JCPU_ONLYY?2	????b 