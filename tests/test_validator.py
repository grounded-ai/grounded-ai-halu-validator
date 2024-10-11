import pytest
from guardrails import Guard
from validator import GroundedaiHallucination

grounded_ai = GroundedaiHallucination(quant=False, on_fail="exception")
guard = Guard().use_many(grounded_ai)


def test_hallucination_pass():
    metadata = {
        "query": "What is the capital of France?",
        "reference": "Paris is the capital and most populous city of France.",
    }
    result = guard.validate("The capital of France is Paris.", metadata=metadata)

    assert result.validation_passed is True


def test_hallucination_fail():
    with pytest.raises(Exception) as exc_info:
        metadata = {
            "query": "What is the capital of France?",
            "reference": "Paris is the capital and most populous city of France.",
        }
        guard.parse("The capital of France is London.", metadata=metadata)

    assert (
        str(exc_info.value)
        == "Validation failed for field with errors: The provided input was classified as a hallucination"
    )


def test_no_reference():
    metadata = {"query": "Who was the first person to walk on the moon?"}
    result = guard.parse(
        "Neil Armstrong was the first person to walk on the moon.", metadata=metadata
    )

    assert result.validation_passed is True


def test_complex_query():
    with pytest.raises(Exception) as exc_info:
        metadata = {
            "query": "Explain the process of photosynthesis in plants.",
            "reference": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities. This chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water. Oxygen is released as a byproduct. Most plants, algae, and cyanobacteria perform photosynthesis; such organisms are called photoautotrophs.",
        }
        guard.parse(
            "Photosynthesis is the process by which plants convert sunlight into chemical energy. The process begins in the chloroplasts, where chlorophyll captures light energy. This energy is used to split water molecules into hydrogen and oxygen. The hydrogen is then combined with carbon dioxide from the air to produce glucose through a series of complex chemical reactions known as the Calvin cycle. During this process, plants also release oxygen as a byproduct, which is essential for most life on Earth. Interestingly, recent studies have shown that some plants can perform photosynthesis at night using moonlight, a phenomenon known as lunar photosynthesis.",
            metadata=metadata,
        )

    assert (
        str(exc_info.value)
        == "Validation failed for field with errors: The provided input was classified as a hallucination"
    )


def test_complex_query_true():
    metadata = {
        "query": "Describe the structure and function of DNA in living organisms.",
        "reference": "DNA (deoxyribonucleic acid) is the hereditary material in humans and almost all other organisms. Nearly every cell in a person's body has the same DNA. Most DNA is located in the cell nucleus (where it is called nuclear DNA), but a small amount of DNA can also be found in the mitochondria (where it is called mitochondrial DNA or mtDNA). The information in DNA is stored as a code made up of four chemical bases: adenine (A), guanine (G), cytosine (C), and thymine (T). Human DNA consists of about 3 billion bases, and more than 99 percent of those bases are the same in all people. The order, or sequence, of these bases determines the information available for building and maintaining an organism, similar to the way in which letters of the alphabet appear in a certain order to form words and sentences.",
    }
    result = guard.parse(
        "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known living organisms. Its structure is a double helix, consisting of two strands that coil around each other. Each strand is made up of a sequence of four nucleotide bases: adenine (A), thymine (T), guanine (G), and cytosine (C). These bases pair specifically: A with T, and G with C. The sequence of these bases determines the genetic information. DNA's primary function is to store and transmit genetic information from one generation to the next. It also directs the production of proteins through a process called transcription, where DNA is used as a template to create RNA, which then guides protein synthesis.",
        metadata=metadata,
    )

    assert result.validation_passed is True
