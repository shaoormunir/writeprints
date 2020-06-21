from writeprints.text_processor import Processor

processor = Processor(False)

output = processor.process("This is a sample test and I am going to look at it")

print(len(output.keys()))