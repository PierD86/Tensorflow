	???9#?M@???9#?M@!???9#?M@	+??5돲?+??5돲?!+??5돲?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???9#?M@??d?`T??Ar?????M@Y??_vO??*	fffff?P@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?+e?X??!???(?@@)+??????1??؉??<@:Preprocessing2F
Iterator::ModelF%u???!F??
цC@)???&??19?Z$??;@:Preprocessing2U
Iterator::Model::ParallelMapV2? ?	??!?B????&@)? ?	??1?B????&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??0?*??!7|???t1@)y?&1?|?1?<pƵ$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0?*??!?}e?.yN@)a2U0*?s?1?wɃg@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*?s?!?wɃg@)a2U0*?s?1?wɃg@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF%u?k?!F??
ц@)F%u?k?1F??
ц@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_?Qڋ?!o_Y?K4@)??H?}]?1??"AM@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9+??5돲?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??d?`T????d?`T??!??d?`T??      ??!       "      ??!       *      ??!       2	r?????M@r?????M@!r?????M@:      ??!       B      ??!       J	??_vO????_vO??!??_vO??R      ??!       Z	??_vO????_vO??!??_vO??JCPU_ONLYY+??5돲?b 