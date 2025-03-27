"""
This code was adapted from the original version to improve readability and maintainability.
Original code: https://github.com/jayralencar/check-eval/blob/main/checkeval/checkeval.py
"""
# Import necessary modules and classes
from phi3 import Phi3Client  # Custom client for interacting with the Phi3 service
import json  # For handling JSON data
from typing import List, Optional  # For type annotations
from pydantic import BaseModel, Field, conint  # For data validation and modeling
from pathlib import Path  # For handling file paths

# Default definitions for evaluation criteria
default_criterion_definitions = {
    "consistency": "the factual alignment between the candidate and the reference text. A factually consistent candidate contains only statements that are entailed by the reference document.",
    "coherence": "Coherence refers to the overall quality that ensures sentences in a text build logically from one to the next, forming a well-structured and well-organized body of information on a given topic.",
    "relevance": "selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries that contained redundancies and excess information.",
    "fluency": "the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.",
}

# Define data models using Pydantic for data validation and parsing


class ChecklistItem(BaseModel):
    number: conint(ge=1) = Field(
        ..., description="The item number"
    )  # Item number, must be >=1
    text: str = Field(
        ..., description="The text of the checklist item"
    )  # Description of the checklist item


class Checklist(BaseModel):
    items: List[ChecklistItem] = Field(
        ..., description="The checklist items"
    )  # List of checklist items

    def to_markdown(self) -> str:
        """Convert the checklist to Markdown format."""
        markdown = "# Checklist\n"
        for item in self.items:
            markdown += f"{item.number}. {item.text}\n"
        return markdown


class ChecklistResponseItem(BaseModel):
    item: conint(ge=1) = Field(
        ..., description="Identifier for the checklist item."
    )  # Reference to checklist item number
    isChecked: bool = Field(
        ..., description="Indicates if the candidate contemplates the item."
    )  # Whether the candidate meets the checklist item


class ChecklistResponse(BaseModel):
    items: List[ChecklistResponseItem] = Field(
        ..., description="List of individual checklist item responses."
    )  # List of responses corresponding to checklist items

    def call(self):
        """Convert response items to a list of dictionaries."""
        return [
            {"item": item.item, "contemplated": item.isChecked} for item in self.items
        ]

    def score(self) -> float:
        """Calculate the overall score based on checked items."""
        if not self.items:
            return 0.0  # Avoid division by zero
        return sum(item.isChecked for item in self.items) / len(
            self.items
        )  # Proportion of items checked


class Checkeval:
    def __init__(self, criteria_definitions: Optional[dict] = None):
        """
        Initialize the Checkeval instance with optional criteria definitions.

        Args:
            criteria_definitions (Optional[dict]): Custom definitions for evaluation criteria.
        """
        self.client = Phi3Client()  # Initialize the Phi3 client
        self.criteria_definitions = (
            criteria_definitions or default_criterion_definitions
        )  # Use provided definitions or default ones

    def call_model(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        functions: Optional[dict] = None,
    ) -> str:
        """
        Call the Phi3Client with the provided messages.

        Args:
            messages (List[dict]): A list of messages with 'role' and 'content'.
            tools (Optional[List[dict]]): Tools to assist the model (unused in current implementation).
            functions (Optional[dict]): Functions to extend model capabilities (unused in current implementation).

        Returns:
            str: The response text from the Phi3Client.
        """
        user_prompt = ""  # Initialize user prompt
        system_prompt = ""  # Initialize system prompt
        for message in messages:
            role = message["role"]  # Role of the message ('system' or 'user')
            content = message["content"]  # Content of the message
            if role == "system":
                system_prompt += f"{content}\n"  # Append to system prompt
            else:
                user_prompt += f"{content}\n"  # Append to user prompt

        if system_prompt:
            self.client.system_prompt = (
                system_prompt.strip()
            )  # Set system prompt in the client

        # Debugging: Print the system prompt and user prompt
        print("\n--- System Prompt ---")
        print(self.client.system_prompt)
        print("\n--- User Prompt ---")
        print(user_prompt.strip())

        # Call the Phi3Client with the concatenated user prompt
        return self.client.get_response(
            user_prompt.strip(), verbose=True
        )  # verbose=True for detailed logs

    def generate_checklist(
        self,
        criterion: str,
        text: Optional[str] = None,
        prompt: Optional[str] = None,
        criterion_definition: Optional[str] = None,
    ) -> Optional[Checklist]:
        """
        Generate a checklist based on the given criterion.

        Args:
            criterion (str): The evaluation criterion (e.g., 'consistency').
            text (Optional[str]): Reference text to base the checklist on.
            prompt (Optional[str]): Custom prompt for generating the checklist.
            criterion_definition (Optional[str]): Definition of the criterion.

        Returns:
            Optional[Checklist]: The generated checklist or None if failed.
        """
        # Use provided criterion definition or default
        criterion_definition = criterion_definition or self.criteria_definitions.get(
            criterion, "No definition provided."
        )

        if prompt is None:
            # Load the prompt from the 'generate_checklist.md' file
            prompt_path = Path(__file__).parent / "prompts" / "generate_checklist.md"
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt = file.read()
            # Format the prompt with the criterion and its definition
            prompt = prompt.format(
                criterion=criterion, criterion_definition=criterion_definition
            )

        # Instruct the assistant to output in JSON format
        prompt += '\n\nPlease output the checklist in the following JSON format:\n```json\n{\n  "items": [\n    {"number": 1, "text": "First item"},\n    {"number": 2, "text": "Second item"},\n    ...\n  ]\n}\n```\n'

        # Prepare messages for the model
        messages = [{"role": "system", "content": prompt}]
        if text:
            # If reference text is provided, include it as user content
            messages.append({"role": "user", "content": f"### Reference text\n{text}"})

        # Call the model to generate the checklist
        response_text = self.call_model(messages)

        # Debugging: Print the response text
        print("\n--- Response Text (Generate Checklist) ---")
        print(response_text)

        try:
            # Extract JSON from the assistant's response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            response_json = json.loads(json_str)  # Parse the JSON string
            return Checklist(**response_json)  # Return a Checklist object
        except Exception as e:
            # Handle parsing errors
            print("Error parsing response:", e)
            print("Response text:", response_text)
            return None

    def evaluate_checklist(
        self,
        text: str,
        checklist: Checklist,
        criterion: str,
        prompt: Optional[str] = None,
        criterion_definition: Optional[str] = None,
    ) -> Optional[ChecklistResponse]:
        """
        Evaluate the candidate text against the checklist.

        Args:
            text (str): The candidate text to evaluate.
            checklist (Checklist): The checklist to use for evaluation.
            criterion (str): The evaluation criterion.
            prompt (Optional[str]): Custom prompt for evaluation.
            criterion_definition (Optional[str]): Definition of the criterion.

        Returns:
            Optional[ChecklistResponse]: The evaluation results or None if failed.
        """
        # Use provided criterion definition or default
        criterion_definition = criterion_definition or self.criteria_definitions.get(
            criterion, "No definition provided."
        )

        if prompt is None:
            # Load the prompt from the 'evaluate_checklist.md' file
            prompt_path = Path(__file__).parent / "prompts" / "evaluate_checklist.md"
            with open(prompt_path, "r", encoding="utf-8") as file:
                prompt = file.read()

            # Convert checklist to Markdown format for inclusion in the prompt
            checklist_md = (
                checklist.to_markdown()
                if isinstance(checklist, Checklist)
                else str(checklist)
            )
            # Format the prompt with the criterion details and checklist
            prompt = prompt.format(
                criterion=criterion,
                criterion_definition=criterion_definition,
                checklist=checklist_md,
            )

        # Instruct the assistant to output in JSON format
        prompt += '\n\nPlease output your evaluation in the following JSON format:\n```json\n{\n  "items": [\n    {"item": 1, "isChecked": true},\n    {"item": 2, "isChecked": false},\n    ...\n  ]\n}\n```\n'

        # Prepare messages for the model
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"### Candidate text\n{text}"},
        ]

        # Call the model to evaluate the checklist against the candidate text
        response_text = self.call_model(messages)

        # Debugging: Print the response text
        print("\n--- Response Text (Evaluate Checklist) ---")
        print(response_text)

        try:
            # Extract JSON from the assistant's response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            response_json = json.loads(json_str)  # Parse the JSON string
            return ChecklistResponse(
                **response_json
            )  # Return a ChecklistResponse object
        except Exception as e:
            # Handle parsing errors
            print("Error parsing response:", e)
            print("Response text:", response_text)
            return None

    def reference_guided(
        self,
        criterion: str,
        reference: str,
        candidate: str,
        checklist: Optional[Checklist] = None,
        prompt: Optional[str] = None,
        criterion_definition: Optional[str] = None,
    ) -> dict:
        """
        Perform reference-guided evaluation.

        This method generates a checklist based on the reference text and evaluates the candidate text against it.

        Args:
            criterion (str): The evaluation criterion.
            reference (str): The reference text.
            candidate (str): The candidate text.
            checklist (Optional[Checklist]): Predefined checklist. If None, a new checklist is generated.
            prompt (Optional[str]): Custom prompt for checklist generation.
            criterion_definition (Optional[str]): Definition of the criterion.

        Returns:
            dict: A dictionary containing the checklist and evaluation results.
        """
        if not checklist:
            # Generate a checklist if not provided
            checklist = self.generate_checklist(
                criterion,
                text=reference,
                prompt=prompt,
                criterion_definition=criterion_definition,
            )
            if not checklist:
                # Return None values if checklist generation failed
                return {"checklist": None, "results": None}

        # Evaluate the candidate text against the checklist
        results = self.evaluate_checklist(
            candidate,
            checklist,
            criterion,
            prompt=prompt,
            criterion_definition=criterion_definition,
        )
        return {
            "checklist": checklist,
            "results": results,
        }

    def candidate_guided(
        self,
        criterion: str,
        reference: str,
        candidate: str,
        checklist: Optional[Checklist] = None,
        prompt: Optional[str] = None,
        criterion_definition: Optional[str] = None,
    ) -> dict:
        """
        Perform candidate-guided evaluation by swapping reference and candidate.

        This method treats the candidate text as the reference and vice versa.

        Args:
            criterion (str): The evaluation criterion.
            reference (str): The reference text.
            candidate (str): The candidate text.
            checklist (Optional[Checklist]): Predefined checklist.
            prompt (Optional[str]): Custom prompt for checklist generation.
            criterion_definition (Optional[str]): Definition of the criterion.

        Returns:
            dict: A dictionary containing the checklist and evaluation results.
        """
        # Swap reference and candidate texts
        return self.reference_guided(
            criterion, candidate, reference, checklist, prompt, criterion_definition
        )

    def criterion_guided(
        self,
        criterion: str,
        reference: str,
        candidate: str,
        checklist: Optional[Checklist] = None,
        prompt: Optional[str] = None,
        criterion_definition: Optional[str] = None,
    ) -> dict:
        """
        Perform criterion-guided evaluation using specific prompts.

        This method uses specialized prompts for generating and evaluating the checklist based on the criterion.

        Args:
            criterion (str): The evaluation criterion.
            reference (str): The reference text.
            candidate (str): The candidate text.
            checklist (Optional[Checklist]): Predefined checklist.
            prompt (Optional[str]): Custom prompt for checklist generation.
            criterion_definition (Optional[str]): Definition of the criterion.

        Returns:
            dict: A dictionary containing the checklist and evaluation results.
        """
        # Load the generate prompt from 'criterion_generate.md'
        generate_prompt_path = (
            Path(__file__).parent / "prompts" / "criterion_generate.md"
        )
        with open(generate_prompt_path, "r", encoding="utf-8") as file:
            generate_prompt = file.read()

        # Use provided criterion definition or default
        criterion_definition = criterion_definition or self.criteria_definitions.get(
            criterion, "No definition provided."
        )

        if not checklist:
            # Format the generate prompt with criterion details
            generate_prompt = generate_prompt.format(
                criterion=criterion, criterion_definition=criterion_definition
            )
            # Generate the checklist using the customized prompt
            checklist = self.generate_checklist(
                criterion,
                text=reference,
                prompt=generate_prompt,
                criterion_definition=criterion_definition,
            )
            if not checklist:
                # Return None values if checklist generation failed
                return {"checklist": None, "results": None}

        # Convert the checklist to Markdown format
        checklist_md = (
            checklist.to_markdown()
            if isinstance(checklist, Checklist)
            else str(checklist)
        )

        # Load the evaluate prompt from 'criterion_evaluate.md'
        evaluate_prompt_path = (
            Path(__file__).parent / "prompts" / "criterion_evaluate.md"
        )
        with open(evaluate_prompt_path, "r", encoding="utf-8") as file:
            evaluate_prompt = file.read()

        # Format the evaluate prompt with criterion details and checklist
        evaluate_prompt = evaluate_prompt.format(
            criterion=criterion,
            criterion_definition=criterion_definition,
            checklist=checklist_md,
        )

        # Instruct the assistant to output in JSON format
        evaluate_prompt += '\n\nPlease output your evaluation in the following JSON format:\n```json\n{\n  "items": [\n    {"item": 1, "isChecked": true},\n    {"item": 2, "isChecked": false},\n    ...\n  ]\n}\n```\n'

        # Prepare messages for the model
        messages = [
            {"role": "system", "content": evaluate_prompt},
            {
                "role": "user",
                "content": f"### Reference text\n{reference}\n\n### Candidate text\n{candidate}",
            },
        ]

        # Call the model to evaluate the checklist against the candidate text
        response_text = self.call_model(messages)

        # Debugging: Print the response text
        print("\n--- Response Text (Criterion Guided Evaluation) ---")
        print(response_text)

        try:
            # Extract JSON from the assistant's response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            response_json = json.loads(json_str)  # Parse the JSON string
            return {
                "checklist": checklist,
                "results": ChecklistResponse(**response_json),
            }
        except Exception as e:
            # Handle parsing errors
            print("Error parsing response:", e)
            print("Response text:", response_text)
            return {
                "checklist": checklist,
                "results": None,
            }


# Main execution block
if __name__ == "__main__":
    # Initialize the Checkeval instance
    checkeval = Checkeval()

    # Define the evaluation criterion
    criterion = "consistency"

    # Reference text (ground truth)
    reference_text = """
    As introduced in Section 3, MIDI is a descriptive “music language", which describes the music information to be performed in bytes, such as what instrument to use, what note to start with, and what note to end at a certain time. MIDI can be employed to listen or input into the analysis program that only requires the basic music description of music score. The MIDI file itself does not contain waveform data, so the file is very small. The pretty_midi Python toolkit contains practical functions/classes for parsing, modifying and processing MIDI data, through which users can easily read various note information contained in MIDI.   Music21 is an object-oriented toolkit for analyzing, searching and converting music in symbolic form. J. S. Bach four-part chorus dataset can be directly obtained from music21 Python package, which contains 402 choruses. The four parts in the dataset are soprano, alto, tenor and bass. However, this data set is very small and lacks expressive information.   Ferreira et al. created a new music dataset VGMIDI with sentiment notation in symbolic format, which contains 95 MIDI labelled piano pieces (966 phrases of 4 bars) from video game soundtracks and 728 non-labelled pieces, all of them vary in length from 26 seconds to 3 minutes. MIDI labelled music pieces is annotated by 30 human subjects according to a valence-arousal (dimensional) model of emotion. The sentiment of each piece is then extracted by summarizing the 30 annotations and mapping the valence axis to sentiment. For the concrete operation of emotion annotation extraction, please refer to literature .   The Lakh MIDI Dataset (LMD) is the largest symbolic music corpus to date, including 176,581 unique MIDI files created by Colin Raffel, of which 45,129 files have been matched and aligned with the items in the Million Song Dataset (MSD) BIBREF356 . However, the dataset has unlimited polyphonic, inconsistent expressive characteristics and contains various genres, instruments and time periods. LMD includes the following formats: 1) 176,581 MIDI files with duplicate data removed, and each file is named according to its MD5 checksum (called “LMD full"); 2) subset of 45,129 files (called “LMD matched") that match items in the MSD; 3) All LMD-matched files are aligned with the 7Digital preview MP3s in the MSD (called “LMD aligned").  5pt |m50pt|m80pt|m27pt|m30pt|m27pt|m55pt|m120pt|m110pt|  Dataset summary 2* Format 2* Name 3c| Modality 2*   2* Size 2* Access   Score   Audio  8rContinued table 2 2* Format 2* Name 3c| Modality 2*   2* Size 2* Access   Score   Audio  10*MIDI JSB Chorus Polyphonic 402 Bach four parts chorus Music21toolkit BIBREF312   VGMIDI Polyphonic with sentiment 823 piano video game soundtracks Derived from BIBREF306   Lakh MIDI Dataset Multi-instrumental 176,581MIDI files http://colinraffel.com/pro-jects/lmd/  Projective Orchestral Database Orchestral 392 MIDI files grouped in pairs containing a piano score and its orchestral version https://qsdfo.github.io/LOP-/database  e-Piano Competition Dataset Polyphonic & Performance \\sim 1400 MIDI files of piano performance http://www.piano-e-competition.com  BitMidi Polyphonic 113,244 MIDI files curated by volunteers around the world https://bitmidi.com/  Classical Archives Polyphonic Maximum number of MIDI files of free classical music https://www.classical-archives.com/  The largest MIDI dataset on the Internet Polyphonic & Style About 130,000 pieces of music from 8 distinct genres (classical, metal, folk, etc.) http://stoneyroads.com/20-15/06/behold-the-worlds-biggest-midicollection-on-the-internet/  ADL Piano MIDI Polyphonic 11,086 unique piano MIDI files https://github.com/lucasnfe/-adl-piano-midi  GiantMIDI-Piano Polyphonic 10,854 MIDI files of classical piano, 1,237 hours in total https://github.com/byte-dance/GiantMIDI-Piano 4*MusicXML TheoryTab Database Polyphonic 16K lead sheet segments https://www.hooktheory.-com/theorytab  Hooktheory Lead Sheet dataset Polyphonic 11,329 lead sheet segments Derived from BIBREF179   Wikifonia Polyphonic 2,252 western music lead sheets http://marg.snu.ac.kr/chord_-generation/(CSV format)  MuseScore lead sheet dataset Performance lead sheet corresponding to Yamaha e-Competitions MIDI dataset https://musescore.com Pianoroll Lakh Pianoroll Dataset Multi-instrumental Approximately equal to the size of LMD https://salu133445.github.-io/musegan/ 4*Text Nottingham Music Dataset Monophonic About 1,000 folk songs abc.sourceforge.net/NMD/  ABC tune book of Henrik Norbeck Monophonic More than 2,800 scores and lyrics in ABC format, mainly Irish and Swiss traditional music http://www.norbeck.nu/abc/  ABC version of FolkDB Monophonic Unknown https://thesession.org/  KernScores Polyphonic Over 700 million notes in 108,703 files http://kern.humdrum.org 6*Audio NSynth Dataset Music audio 306,043 notes https://magenta.tensorflow.-org/datasets/nsynth  FMA dataset Music audio 106,574 tracks of 917GiB Derived from cite212  Minist musical sound dataset Music audio 50,912 notes https://github.com/ejhum-phrey/minst-dataset/  GTZAN Dataset Music audio 1,000 30s music audios http://marsyas.info/down-load/data_sets  Studio On-Line (SOL) Music audio 120,000 sounds Derived from BIBREF315   NUS Sung and Spoken Lyrics(NUS-48E) Corpus Sing Voice 169 minutes recordings of 48 English songs Derived from BIBREF316  9*  MusicNet Dataset Fusion 330 recordings of classical music https://homes.cs.washing-ton.edu/\\sim thickstn/musicnet-.html  MAESTRO Dataset Fusion 172 hours of virtuosic piano performances https://g.co/magenta/-maestrodataset  NES Music Database Multi-instrumental thousands of Derived from BIBREF199   Piano-Midi Polyphonic & performance 332 classical piano pieces www.piano-midi.de/  Groove MIDI Dataset Drum 13.6 hours recordings, 1,150 MIDI files and over 22,000 measures of tempo-aligned expressive drumming https://magenta.tensorflow-.org/datasets/groove  POP909 Polyphonic multiple versions of the piano arrangements of 909 popular songs https://github.com/music-x-lab/POP909-Dataset  ASAP Polyphonic Performance& Fusion 222 digital musical scores aligned with 1,068 performances https://github.com/fosfrance-sco/asap-dataset  Aligned lyrics-melody music dataset Fusion 13,937 20-note sequences with 278,740 syllable-note pairs https://github.com/yy1lab/-Lyrics-Conditioned-Neural-Melody-Generation  MTM Dataset Fusion Unknown https://github.com/Morning-Books/MTM-Dataset  The Projective Orchestral Database (POD) is devoted to the study of the relationship between piano scores and corresponding orchestral arrangements. It contains 392 MIDI files, which are grouped in pairs containing a piano score and its orchestral version. In order to facilitate the research work, crestel et al. BIBREF357 provided a pre-computed pianoroll representation. In addition, they also proposed a method to automatically align piano scores and their corresponding orchestral arrangements, resulting in a new version of MIDI database. They provide all MIDI files as well as preprocessed pianoroll representations of alignment and misalignment for free on the following website https://qsdfo.github.io/LOP/index.html.   The e-piano junior competition is an international classical piano competition. The e-piano junior competition dataset is a collection of professional pianists\' solo piano performances. It is the largest public dataset that provides a substantial amount of expressive performance MIDI ( 1400) of professional pianists. Most of them are late romantic works, such as Chopin and Liszt, as well as some Mozart sonatas. Since this dataset provides high-quality piano performance data in MIDI, including the fine control of timing and dynamics by different performers, the dataset is widely used in the research of performance generation, but it does not contain the corresponding music score of the pieces BIBREF354 .   The ADL piano MIDI dataset is based on LMD. In LMD, there are many versions of the same song, and only one version is reserved for each song in the ADL dataset. Later, Ferreira et al. BIBREF355 extracted from the LMD only the tracks with instruments from the “piano family" (MIDI program number 1-8). This process generated a total of 9,021 unique piano MIDI files. These files are mainly rock and classical music, so in order to increase the genres diversity (as jazz, etc.) of the dataset, they have added another 2,065 files obtained from public resources on the Internet FOOTREF78 . All the files in the collection are de-duped according to MD5 checksums, and the final dataset has 11,086 pieces.   Recently, ByteDance released GiantMIDI-Piano BIBREF358, the world\'s largest classical piano dataset, including MIDI files from 10,854 music works of 2,784 composers, with a total duration of 1,237 hours. In terms of data scale, the total duration of different music pieces in the dataset is 14 times that of Google’s MAESTRO dataset. In order to construct the dataset, researchers have developed and open-sourced a high-resolution piano transcription system, which is used to convert all audio into MIDI files. MIDI files include the onset, dynamics and pedal information of notes.   In addition, BitMidi FOOTREF79 provides 113,244 MIDI files curated by volunteers around the world; Classical Archives FOOTREF80 is the largest classical music website, including the largest collection of free classical music MIDI files; the largest MIDI dataset FOOTREF81 on the Internet contains about 130,000 music from eight different genres (classical, metal, folk, etc.); FreeMidi FOOTREF82 comprises more than 25,860 MIDI files of assorted genres.
    """

    # Candidate text (generated summary)
    candidate_text = """
    The exploration of datasets in deep music generation, particularly focusing on MIDI representations, reveals significant advancements and insights that enhance the field. The comparative analysis of automatic melody harmonization models, notably the MTHarmonizer, demonstrates its superior performance in generating diverse and interesting chord progressions compared to traditional methods like BiLSTM and HMM. The robustness of the findings is supported by a substantial dataset of 9,226 melody/chord pairs, which not only facilitates comprehensive model evaluation across various musical contexts but also mitigates overfitting, thereby improving the reliability of generated harmonizations [BIBREF179].\n\nThe NES Music Database (NES-MDB) further enriches the understanding of expressive performance characteristics by providing a dual focus on composition and performance. This dataset includes detailed expressive attributes related to dynamics and timbre, which are crucial for generating realistic musical renditions. The tool associated with NES-MDB enables the rendering of compositions into NES-style audio, enhancing the practical application of the dataset in music generation research [BIBREF199]. The establishment of baseline results within NES-MDB underscores the importance of expressive performance in music generation, indicating a shift towards models that integrate these characteristics for improved musical outputs [BIBREF199].\n\nThe GiantMIDI-Piano dataset stands out as the largest piano MIDI dataset, offering extensive statistical analyses that facilitate deeper musical insights. Its meticulous curation process and the inclusion of live performance MIDI files enhance the accuracy of piano solo detection and transcription, marking a significant improvement over previous datasets. The dataset's comprehensive nature allows for various applications in music information retrieval, including computer-based musical analysis and expressive performance analysis, thus serving as a valuable resource for advancing research in these areas [BIBREF358].\n\nAdditionally, the Projective Orchestral Database (POD) introduces a novel framework for automatic orchestration by linking piano scores with their orchestral counterparts. This task enables researchers to explore the correlations between different musical representations, paving the way for advancements in orchestration techniques through learning-based models [BIBREF357]. The methodologies employed in the POD highlight the importance of statistical learning methods in capturing the dependencies between piano and orchestral scores, which is crucial for enhancing automatic orchestration processes [BIBREF357].\n\nOverall, the integration of these datasets into deep music generation research not only enhances the understanding of musical structures and expressive characteristics but also informs the development of more sophisticated algorithms capable of generating realistic and emotionally resonant music. Future directions in this field may involve refining these datasets, expanding their diversity, and improving the methodologies used for music generation, ultimately leading to richer and more nuanced musical experiences.
    """

    # -------------------------------
    # Reference-Guided Evaluation
    # -------------------------------

    # Generate the checklist and evaluate using reference_guided method
    result_ref = checkeval.reference_guided(criterion, reference_text, candidate_text)

    # Print the generated checklist from reference-guided evaluation
    print("\nGenerated Checklist (Reference Guided):")
    if result_ref["checklist"]:
        for item in result_ref["checklist"].items:
            print(f"{item.number}. {item.text}")
    else:
        print("Failed to generate checklist.")

    # Print the evaluation results from reference-guided evaluation
    print("\nEvaluation Results (Reference Guided):")
    if result_ref["results"]:
        for item in result_ref["results"].items:
            status = (
                "Yes" if item.isChecked else "No"
            )  # Interpret boolean as 'Yes' or 'No'
            print(f"Item {item.item}: {status}")
        print(
            f"\nOverall Score: {result_ref['results'].score()*100:.2f}%"
        )  # Print overall percentage
    else:
        print("Failed to evaluate candidate text.")

    # -------------------------------
    # Criterion-Guided Evaluation
    # -------------------------------

    # Generate the checklist and evaluate using criterion_guided method
    result_crit = checkeval.criterion_guided(criterion, reference_text, candidate_text)

    # Print the generated checklist from criterion-guided evaluation
    print("\nGenerated Checklist (Criterion Guided):")
    if result_crit["checklist"]:
        if isinstance(result_crit["checklist"], Checklist):
            for item in result_crit["checklist"].items:
                print(f"{item.number}. {item.text}")
        else:
            # In case checklist is not a Checklist object, print as is
            print(result_crit["checklist"])
    else:
        print("Failed to generate checklist.")

    # Print the evaluation results from criterion-guided evaluation
    print("\nEvaluation Results (Criterion Guided):")
    if result_crit["results"]:
        for item in result_crit["results"].items:
            status = (
                "Yes" if item.isChecked else "No"
            )  # Interpret boolean as 'Yes' or 'No'
            print(f"Item {item.item}: {status}")
        print(
            f"\nOverall Score: {result_crit['results'].score()*100:.2f}%"
        )  # Print overall percentage
    else:
        print("Failed to evaluate candidate text.")
