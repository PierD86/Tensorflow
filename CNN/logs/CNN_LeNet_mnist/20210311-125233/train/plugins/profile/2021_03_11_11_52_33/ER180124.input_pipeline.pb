	??C?l7H@??C?l7H@!??C?l7H@	???f%?????f%??!???f%??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??C?l7H@?|?5^???A??MbH@Y?$??C??*	?????a@2F
Iterator::Modelŏ1w-??!c?1ƘH@)HP?sע?1?ꫯ??:@:Preprocessing2U
Iterator::Model::ParallelMapV2?sF????!7?l??66@)?sF????17?l??66@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???߾??!Zh??4@)???߾??1Zh??4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz?,C??!<??<4@)?b?=y??1?Zh??1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate7?[ A??!??N;??8@) ?o_?y?1z??g?y@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipv??????!??s?9gI@)a??+ey?1/?袋.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???_vOn?!??6?l?@)???_vOn?1??6?l?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V-??!l??:@)??H?}]?1u?QG??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???f%??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?|?5^????|?5^???!?|?5^???      ??!       "      ??!       *      ??!       2	??MbH@??MbH@!??MbH@:      ??!       B      ??!       J	?$??C???$??C??!?$??C??R      ??!       Z	?$??C???$??C??!?$??C??JCPU_ONLYY???f%??b 