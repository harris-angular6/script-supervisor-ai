import json
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

USE_RAG = True
RAG_CONFIDENCE_THRESHOLD = 0.60
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

PROJECT_ROOT = Path(".")
JSON_INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes"
JSON_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "scenes"

class RagContinuityDetector:
    def __init__(self, input_jsonl_name, output_jsonl_name, model_name):
        self.enabled = USE_RAG
        self.input_file_name = input_jsonl_name
        self.output_file_name = output_jsonl_name
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="script_supervisor_scenes")
  
    def BuildSceneDocumentText(self, scene):
        return f"""
            Slugline: {scene.get("slugline", "")}
            Location: {scene.get("location", "")}
            Time of day: {scene.get("time_of_day", "")}
            Characters: {", ".join(scene.get("characters", []))}
            Props: {", ".join(scene.get("props", []))}
            Actions: {" ".join(scene.get("actions", []))}

            Scene text:
            {scene.get("scene_text", "")}
        """.strip()

    def GetSceneList(self):
        scene = None
        scene_document_text = None
        metadata = None
        
        with open(self.input_file_name, "r", encoding="utf-8") as jsonl_file:
            for json_line in jsonl_file:
                scene = json.loads(json_line)
                scene_document_text = self.BuildSceneDocumentText(scene)

                metadata = {
                    "script_id": scene["script_id"],
                    "scene_id": scene["scene_id"],
                    "scene_index": scene["scene_index"],
                    "slugline": scene.get("slugline", ""),
                    "location": scene.get("location", ""),
                    "time_of_day": scene.get("time_of_day", "")
                }

        return scene, scene_document_text, metadata

    def LoadScenes(self, jsonl_file_path: str):
        scenes = []
        
        with open(jsonl_file_path, "r", encoding="utf-8") as file_movie_scenes:
            for i, json_line in enumerate(file_movie_scenes):
            #for scene in file_movie_scenes:
                scene = json.loads(json_line)                
                scenes.append(scene)               
                
                if i >= 30:
                    break;

        return scenes

    def GetStronglyRelatedScenes(self, current_scene: dict, collection, top_k: int = 10, max_distance: float = 0.8):
        
        #print("Current Scene: ", current_scene)
        #print("Collection: ", collection.peek())
        query_text = self.BuildSceneDocumentText(current_scene)
        #print("Query Text: ", query_text)
        
        results = collection.query(query_texts=[query_text],
                                   n_results=top_k,
                                   where={
                                       "$and": [
                                           {"script_id": {"$eq": current_scene["script_id"]}},
                                           {"scene_index": {"$lt": current_scene["scene_index"]}}
                                       ]
                                   }
                                  )

        #print("Result of Query: ", results)
        
        strong_scenes = []

        
        for i in range(len(results["ids"][0])):
            distance_tuple = results["distances"][0][i]
            scene_id_tuple = results["ids"][0][i]
            document_tuple = results["documents"][0][i]
            metadata_tuple = results["metadatas"][0][i]

            scene_id = scene_id_tuple
            #print("Type of scene id: ", type(scene_id))
            #print("Scene id: ", scene_id)
            documents = document_tuple
            #print("Type of document: ", type(documents))
            #print("Document: ", documents)
            metadata = metadata_tuple
            #print("Type of metadata: ", type(metadata))
            #print("Metadata: ", metadata)
            
            #print("Type of distance: ", type(distance))
            #print("Distance: ", distance)

            #distance = float(distance_tuple[0])
            distance = distance_tuple
            #print("Type of distance: ", type(distance))
            #print("Distance: ", distance)

            if distance <= max_distance:
                strong_scenes.append({
                    "scene_id": scene_id,
                    "document": documents,
                    "metadata": metadata,
                    "distance": distance})

            '''
            if distance <= max_distance:
                strong_scenes.append({
                    "scene_id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance})
            '''

        #print("Strong Scenes: ", strong_scenes)
        return strong_scenes                    

    #def SaveSceneToRAGDatabase(self, jsonl_file_path: str):      
    def SaveSceneToRAGDatabase(self, scenes: str): 

        ids = []
        documents = []
        metadatas = []

        #with open(jsonl_file_path, "r", encoding="utf-8") as file_movie_scenes:
            #for json_line in file_movie_scenes:
        #for scene in self.LoadScenes(jsonl_file_path):

        for scene in scenes:
            #scene = json.loads(json_line)
            ids.append(scene["scene_id"])
            documents.append(scene["scene_text"])

            metadatas.append({
                "script_id": scene["script_id"],
                "scene_index": scene["scene_index"],
                "location": scene.get("location"),
                "time_of_day": scene.get("time_of_day"),
                "slugline": scene.get("slugline")
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return self.collection
        
        #for scene in scenes:
            
        #self.collection.add(ids=[scene["scene_id"]], documents=[scene_document_text], metadatas=[metadata])

    def BuildContinuityPrompt(self, current_scene: dict, related_scenes: list):
        current_scene_text = self.BuildSceneDocumentText(current_scene)

        related_scene_text = "\n\n".join(f"""
                                        Previous Scene ID: {scene["scene_id"]}
                                        Scene Index: {scene["metadata"]["scene_index"]}
                                        Similarity Distance: {scene["distance"]}

                                        {scene["document"]}
                                        """.strip()
                                        for scene in related_scenes)

        prompt = f"""
                You are a movie script supervisor ID.

                Your task is to detect continuity errors between the CURRENT SCENE
                and the RELATED PREVIOUS SCENES.

                Check only these continuity categories:
                - character continuity
                - prop continuity
                - wardrobe continuity
                - location continuity
                - time-of-day continuity
                - action continuity

                CURRENT SCENE:
                {current_scene_text}

                RELATED PREVIOUS SCENES:
                {related_scene_text}

                Important rules:
                - Only report an error if there is clear evidence
                - Do not guess.
                - Compare the current scene against previous scenes.
                - Return JSON only.

                Return this JSON format:

                {{
                    "current_scene_id": "{current_scene["scene_id"]}",
                    "has_continuity_error": true,
                    "errors": [
                        {{
                            "error_type": "prop_continuity",
                            "related_previous_scene_id": "scene id here",
                            "description": "Explain the continuity issue.",
                            "evidence_from_previous_scene": "Evidence from previous scene.",
                            "evidence_from_current_scene": "Evidence from current scene.",
                            "confidence": 0.85
                        }}
                    ]
                }}
                """
        return prompt.strip()

    def SendPromptToModel(self, prompt):
        model_name = self.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                    device_map="auto",
                                                    torch_dtype=torch.float)  

        llm_pipeline = pipeline("text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                max_new_tokens=800,
                                temperature=0.2,
                                do_sample=True,
                                return_full_text=False)

        #return llm_pipeline
        output = llm_pipeline(prompt)

        return output

    # def GetStronglyRelatedScenes(current_scene: dict, collection, top_k: int = 10, max_distance: float = 0.8):
    def CheckContinuityWithLLM(self, current_scene, collection):
        related_scenes = self.GetStronglyRelatedScenes(current_scene=current_scene,
                                                       collection=collection,
                                                       top_k=10,
                                                       #top_k=20,
                                                       max_distance=0.8)

        if not related_scenes:
            return {
                "current_scene_id": current_scene["scene_id"],
                "has_continuity_error": False,
                "errors": [],
                "reason": "No strongly related previous scenes found."
            }
        # def BuildContinuityPrompt(self, current_scene: dict, related_scenes: list):
        prompt = self.BuildContinuityPrompt(current_scene=current_scene,
                                            related_scenes=related_scenes)

        mistral_prompt = f"<s>[INST] {prompt} [/INST]"

        # def SendPromptToModel(self):
        #output = self.SendPromptToModel(mistral_prompt)[0]["generated_text"]
        output = self.SendPromptToModel(mistral_prompt)

        return output