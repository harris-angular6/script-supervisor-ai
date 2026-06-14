import json
import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

USE_RAG = True
RAG_CONFIDENCE_THRESHOLD = 0.60
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "./foundation_model/base_llm/llama-3.1-8b-instruct"

PROJECT_ROOT = Path(".")
JSON_INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes"
JSON_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "scenes"

INPUT_DIR = PROJECT_ROOT / "datasets" / "scenes_enriched_error"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "issues_error_rag"

INPUT_FILES = [
    "train_error_scenes_enriched.jsonl",
    #"val_error_scenes_enriched.jsonl",
    #"test_error_scenes_enriched.jsonl",
]

OUTPUT_FILES = [
    "train_error_issues_rag_review.jsonl",
    #"val_error_issues_rag.jsonl",
    #"test_error_scenes_rag.jsonl",
]

class RagContinuityDetector:
    def __init__(self, input_jsonl_name, output_jsonl_name, model_name):
        self.enabled = USE_RAG
        self.input_file_name = input_jsonl_name
        self.output_file_name = output_jsonl_name
        self.model_name = model_name
        self.model = None
        #self.mode_path = MODEL_PATH
        self._pipeline = None
        #self.client = chromadb.PersistentClient(path="./chroma_db")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="script_supervisor_scenes")

    def _get_pipeline(self):
        if self._pipeline is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                 # <-- this is the setting
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
                #llm_int8_enable_fp32_cpu_offload=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     quantization_config=bnb_config,
                                                     device_map="cuda:0",
                                                     attn_implementation="sdpa")

            self._pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=tokenizer,
                max_new_tokens=800,
                temperature=0.2,
                do_sample=True,                
                return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id)
            
            #self._pipeline = pipeline(
            #    "text-generation",
            #    model=self.model,
            #    tokenizer=tokenizer,
            #    return_full_text=False)
        return self._pipeline
  
    def BuildSceneDocumentText(self, scene):

        print("Type of scene varialbe:", type(scene))
        print(scene)
        
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

        '''
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
        '''

    def BuildSceneForContinuityIssue(self, scene):
        return {
            "script_id": scene.get("script_id"),
            "scene_id": scene.get("scene_id"),            
            "scene_index": scene.get("scene_index"),
            "slugline": scene.get("slugline", ""),
            "location": scene.get("location", ""),
            "time_of_day": scene.get("time_of_day", ""),
            "characters": scene.get("characters", []),
            "props": scene.get("normalized_props", scene.get("props", [])),
            "wardrobe": scene.get("wardrobe", {}),
            "actions": scene.get("actions", []),
            "action_summary": scene.get("action_summary", ""),
            "scene_text": scene.get("scene_text", "")
        }

    def BuildSceneDocumentForContinuityIssue(self, scene):
        #print("\nScene (Build Scene Document: ", scene)
        
        return {
            "script_id": scene.get("script_id"),
            "scene_id": scene.get("scene_id"),            
            "scene_index": scene.get("scene_index"),
            "slugline": scene.get("slugline", ""),
            "location": scene.get("location", ""),
            "time_of_day": scene.get("time_of_day", ""),
            "characters": scene.get("characters", []),
            "props": scene.get("normalized_props", scene.get("props", [])),
            "wardrobe": scene.get("wardrobe", {}),
            "actions": scene.get("actions", []),
            "action_summary": scene.get("action_summary", ""),
            "scene_text": scene.get("scene_text", "")
        }
        
    def GetSceneList(self):
        scene = None
        scene_document_text = None
        metadata = None
        
        with open(self.input_file_name, "r", encoding="utf-8") as jsonl_file:
            for json_line in jsonl_file:
                scene = json.loads(json_line)
                #scene_document_text = self.BuildSceneDocumentText(scene)
                scene_document_text = self.BuildSceneForContinuityIssue(scene)

                metadata = {
                    "script_id": scene["script_id"],
                    "scene_id": scene["scene_id"],
                    "scene_index": scene["scene_index"],
                    "slugline": scene.get("slugline", ""),
                    "location": scene.get("location", ""),
                    "time_of_day": scene.get("time_of_day", "")
                }

        return scene, scene_document_text, metadata



    def GetStronglyRelatedScenes(self, current_scene: dict, collection, top_k: int = 10, max_distance: float = 0.8):
        
        #print("Current Scene: ", current_scene)
        #print("Collection: ", collection.peek())
        #query_text = self.BuildSceneDocumentText(current_scene)
        query_text = self.BuildSceneForContinuityIssue(current_scene)
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

    def LoadScenes(self, jsonl_file_path: str):
        scenes = []
        
        with open(jsonl_file_path, "r", encoding="utf-8") as file_movie_scenes:
            for i, json_line in enumerate(file_movie_scenes):
            #for scene in file_movie_scenes:
                scene = json.loads(json_line)                
                scenes.append(scene)               
                
                if i >= 10:
                    break;

        return scenes

    def LoadScenesFromRuleBasedErrorResult(self, scenes_info, jsonl_file_path):
        continuity_issues_rule_based_result = []

        with open(jsonl_file_path, "r", encoding="utf-8") as json_file_scenes:
            scenes = [json.loads(json_line) for json_line in json_file_scenes]

        i = 0
        for scene_info in scenes_info:            
            result_scene = next(
                (
                    scene for scene in scenes
                    if scene["script_id"] == scene_info["script_id"] 
                        and scene["scene_id"] == scene_info["detectors"]["rule_based"]["errors"]["scene_id"]
                ),
                None
            )

            result_related_scene = next(
                (
                    scene for scene in scenes
                    if scene["script_id"] == scene_info["script_id"]
                        and scene["scene_id"] == scene_info["detectors"]["rule_based"]["errors"]["related_scene_id"]
                ),
                None
            )

            continuity_issues_rule_based_result.append({                
                "continuity_issue_id": scene_info.get("continuity_issue_id", ""),                
                "script_id": scene_info.get("script_id", ""),
                "detectors": {
                    "rule_based": {                        
                        "has_error": (
                            scene_info.get("detectors", {}).get("rule_based", {}).get("has_error", False),
                        ),
                        "errors": {                            
                            "scene_id": result_scene.get("scene_id", ""),                            
                            "scene_index": result_scene.get("scene_index", ""),                            
                            "related_scene_id": result_related_scene.get("scene_id", ""),                            
                            "issue_type": (
                                scene_info.get("detector", {}).get("rule_based", {}).get("error", {}).get("issue_type", "")
                            ),                                                            
                            "severity": (
                                scene_info.get("detector", {}).get("rule_based", {}).get("error", {}).get("severity", "")
                            ),                            
                            "description": (
                                scene_info.get("detector", {}).get("rule_based", {}).get("error", {}).get("description", "")
                            ),                            
                            "evidence": (
                                scene_info.get("detector", {}).get("rule_based", {}).get("error", {}).get("evidence", "")
                            ),                            
                            "confidence": (
                                scene_info.get("detector", {}).get("rule_based", {}).get("error", {}).get("confidence", "")
                            )
                        }                           
                    }, 
                    "rag_llm": {
                        "has_error": False, 
                        "errors": None
                    }, 
                    "fine_tuned_llm": {
                        "has_error": False, 
                        "errors": None                    
                    }
                },
                "scene": {
                    # 06-03-2026 - begin here: result_scene
                    "script_id": result_scene.get("script_id", ""),
                    "scene_id": result_scene.get("scene_id", ""),
                    "scene_index": result_scene.get("scene_index", ""),
                    "slugline": result_scene.get("slugline", ""),
                    "location": result_scene.get("location", ""),
                    "time_of_day": result_scene.get("time_of_day", ""),
                    "scene_text": result_scene.get("scene_text", ""),
                    "characters": result_scene.get("characters", ""),
                    "props": result_scene.get("props", ""),
                    "actions": result_scene.get("actions", ""),
                    "normalized_props": result_scene.get("normalized_props", ""),
                    "wardrobe": result_scene.get("wardrobe", ""),
                    "action_summary": result_scene.get("action_summary", ""),
                    "continuity_links": {
                        "previous_scene_id": result_scene.get("continuity_links", {}).get("previous_scene_id", None),
                        "next_scene_id": result_scene.get("continuity_links", {}).get("next_scene_id", ""),
                        "same_location_as_previous": result_scene.get("continuity_links", {}).get("same_location_as_previous", False),
                        "same_time_of_day_as_previous": result_scene.get("continuity_links", {}).get("same_time_of_day_as_previous", False),
                        "continuous_with_previous": result_scene.get("continuity_links", {}).get("continuous_with_previous", False)                        
                    } 
                },
                "related_scene": {
                    "script_id": result_related_scene.get("script_id", ""),
                    "scene_id": result_related_scene.get("scene_id", ""),
                    "scene_index": result_related_scene.get("scene_index", ""),
                    "slugline": result_related_scene.get("slugline", ""),
                    "location": result_related_scene.get("location", ""),
                    "time_of_day": result_related_scene.get("time_of_day", ""),
                    "scene_text": result_related_scene.get("scene_text", ""),
                    "characters": result_related_scene.get("characters", ""),
                    "props": result_related_scene.get("props", ""),
                    "actions": result_related_scene.get("actions", ""),
                    "normalized_props": result_related_scene.get("normalized_props", ""),
                    "wardrobe": result_related_scene.get("wardrobe", ""),
                    "action_summary": result_related_scene.get("action_summary", ""),
                    "continuity_links": {
                        "previous_scene_id": result_related_scene.get("continuity_links", {}).get("previous_scene_id", None),
                        "next_scene_id": result_related_scene.get("continuity_links", {}).get("next_scene_id", ""),
                        "same_location_as_previous": result_related_scene.get("continuity_links", {}).get("same_location_as_previous", False),
                        "same_time_of_day_as_previous": result_related_scene.get("continuity_links", {}).get("same_time_of_day_as_previous", False),
                        "continuous_with_previous": result_related_scene.get("continuity_links", {}).get("continuous_with_previous", False)                        
                    }         
                },
                "final_merged_result": {
                    "has_error": scene_info.get("final_merged_result", {}).get("has_error", False),                     
                    "issue_type": scene_info.get("final_merged_result", {}).get("issue_type", ""),
                    "severity": scene_info.get("final_merged_result", {}).get("severity", ""),
                    "description": scene_info.get("final_merged_result", {}).get("description", ""),
                    "evidence": scene_info.get("final_merged_result", {}).get("evidence", ""),
                    "confidence": scene_info.get("final_merged_result", {}).get("confidence", ""),
                    "supporting_detectors": scene_info.get("final_merged_result", {}).get("supporting_detectors", "")                    
                }
            })
            
            #print("\n", "*" * 100)            
            #print("The Result Scene: ", result_scene)
            #print("\n", "=" * 100)
            #print("The Result Related Scene: ", result_related_scene)
            i += 1            
            if i >= 10:
                break

        

        
        #print("\nThe Count of Continuity Issue: ", len(continuity_issues_rule_based_result), "\n")
        '''
        for continuity_issue in continuity_issues_rule_based_result:
            print("\n", "=" * 150)           
            print("Continuity Issue: ", continuity_issue)

        '''

        return continuity_issues_rule_based_result
        #return continuity_issues_rule_based_result.sort(key=lambda x: x["continuity_issue_id"])

    def LoadScenesFromRuleBasedResult(self, scenes_info, jsonl_file_path):
        scenes = []
        related_scenes = []

        i = 0
        with open(jsonl_file_path, "r", encoding="utf-8") as json_file_scene:
            current_line = next(json_file_scene, None)

            while current_line:
                next_line = next(json_file_scene, None)

                scene = json.loads(current_line)                
                #print("Scene: ", scene)
                next_scene = (
                    json.loads(next_line)
                    if next_line is not None
                    else None
                )

                for scene_info in scenes_info:
                    #print("Scene Info: ", scene_info)
                    
                    if scene_info["script_id"] == scene["script_id"] and scene_info["detectors"]["rule_based"]["errors"]["scene_id"] == scene["scene_id"]:
                        #print("\nThe value of i when scene info == scene: ", i)
                        #print("\nThe value of continuity issue id: ", scene_info["continuity_issue_id"])
                        #print("\nThe value of scene id: ", scene["scene_id"])
                        scenes.append({
                            "script_id": scene["script_id"],
                            "scene_id": scene["scene_id"],
                            "scene_index": scene["scene_index"],
                            "slugline": scene["slugline"],
                            "location": scene["location"],
                            "time_of_day": scene["time_of_day"],
                            "scene_text": scene["scene_text"],
                            "continuity_links": 
                                { "previous_scene_id": scene["continuity_links"]["previous_scene_id"],
                                  "next_scene_id": next_scene["scene_id"] if next_scene is not None else None,
                                  "same_location_as_previous": scene["continuity_links"]["same_location_as_previous"],
                                  "same_time_of_day_as_previous": scene["continuity_links"]["same_time_of_day_as_previous"],
                                  "continuous_with_previous": scene["continuity_links"]["continuous_with_previous"]
                                }
                        })
                    if scene_info["script_id"] == scene["script_id"] and scene_info["detectors"]["rule_based"]["errors"]["related_scene_id"] == scene["scene_id"]:
                        #print("\nThe value of i when related scene info == scene: ", i)
                        #print("\nThe value of related scene in continuity issue id: ", scene_info["continuity_issue_id"])
                        #print("\nThe value of scene id: ", scene["scene_id"])
                        related_scenes.append({
                            "script_id": scene["script_id"],
                            "scene_id": scene["scene_id"],
                            "scene_index": scene["scene_index"],
                            "slugline": scene["slugline"],
                            "location": scene["location"],
                            "time_of_day": scene["time_of_day"],
                            "scene_text": scene["scene_text"],
                            "continuity_links": 
                                { "previous_scene_id": scene["continuity_links"]["previous_scene_id"],
                                  "next_scene_id": next_scene["scene_id"] if next_scene is not None else None,
                                  "same_location_as_previous": scene["continuity_links"]["same_location_as_previous"],
                                  "same_time_of_day_as_previous": scene["continuity_links"]["same_time_of_day_as_previous"],
                                  "continuous_with_previous": scene["continuity_links"]["continuous_with_previous"]
                                }
                        })

                current_line = next_line
    
                i += 1
                #print("\nThe value of counter i: ", i)
                if i >= 20:
                    break
            
        return scenes, related_scenes

    def GetScenesInfoFromRuleBasedResult(self, jsonl_rule_based_path: str):
        scenes_info = []
        rule_based_issues = []

        with open(jsonl_rule_based_path, "r", encoding="utf-8") as issues:
            for issue in issues:
                continuity_issue_id = issue["continuity_issue_id"]
                script_id = issue["script_id"]
                has_error = issue["detectors"]["rule_based"]["has_error"]
                scene_id = issue["detectors"]["rule_based"]["errors"]["scene_id"]
                scene_index = issue["detectors"]["rule_based"]["errors"]["scene_index"]
                related_scene_id = issue["detector"]["rule_based"]["errors"]["related_scene_id"]

                scenes_info.append({
                    "continuity_issue_id": continuity_issue_id,
                    "script_id": script_id,
                    "has_error": has_error,
                    "scene_id": scene_id,
                    "scene_index": scene_index,
                    "related_scene_id": related_scene_id
                })

        return scenes_info          

    #def SaveSceneToRAGDatabase(self, jsonl_file_path: str):      
    def SaveSceneToRAGDatabase(self, scenes: str): 

        ids = []
        documents = []
        metadatas = []

        #with open(jsonl_file_path, "r", encoding="utf-8") as file_movie_scenes:
            #for json_line in file_movie_scenes:
        #for scene in self.LoadScenes(jsonl_file_path):

        #for scene in scenes:

        #self.client.delete_collection(name="script_supervisor_scenes")

        #print("\nThe number of sorted scenes: ", len(sorted(scenes, key=lambda s: s["scene_id"])), "\n")

        #print("The count of collection: ", self.collection.count(), "\n")

        #resultCollection = self.collection.get()     
        
        for scene in sorted(scenes, key=lambda s: s["scene_id"]):
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

            #print("\nRag IDS: ", ids, "\n")

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )      

        resultCollection = self.collection.get()

        #print("\nThe number of elements in collection: ", len(resultCollection), "\n")

        '''
        print("\nThe number of elements in collection[ids]: ", len(resultCollection["ids"]), "\n")

        for i in range(len(resultCollection["ids"])):

            print("===================================")
        
            print("Scene ID:")
            print(resultCollection["ids"][i])
        
            print("\nMetadata:")
            print(resultCollection["metadatas"][i])
        
            print("\nDocument:")
            print(resultCollection["documents"][i])
        
            print("===================================")
        '''
        
        return self.collection

    def BuildContinuityScenePairDocument(self, continuity_issue: str):        
        scene = continuity_issue.get("scene", {})
        related_scene = continuity_issue.get("related_scene", {})
    
        return f"""
            CURRENT SCENE:
            slugline: {scene.get("slugline", "")}
            location: {scene.get("location", "")}
            time of day: {scene.get("time_of_day", "")}
            characters: {", ".join(scene.get("characters", []))}
            props: {", ".join(scene.get("props", []))}
            normalized props: {", ".join(scene.get("normalized_props", []))}
            wardrobe: {json.dumps(scene.get("wardrobe", {}), ensure_ascii=False)}
            actions: {" ".join(scene.get("actions", []))}
            action summary: {scene.get("action_summary", "")}
            
            Scene text:
            {scene.get("scene_text", "")}
            
            RELATED SCENE:
            slugline: {related_scene.get("slugline", "")}
            location: {related_scene.get("location", "")}
            time of day: {related_scene.get("time_of_day", "")}
            characters: {", ".join(related_scene.get("characters", []))}
            props: {", ".join(related_scene.get("props", []))}
            normalized props: {", ".join(related_scene.get("normalized_props", []))}
            wardrobe: {json.dumps(related_scene.get("wardrobe", {}), ensure_ascii=False)}
            actions: {" ".join(related_scene.get("actions", []))}
            action summary: {related_scene.get("action_summary", "")}
            
            related scene text:
            {related_scene.get("scene_text", "")}
            """.strip()

    def SaveContinuityIssuesToRAGDatabase(self, continuity_issues: str):
        ids = []
        documents = []
        metadatas = []

        # print("Continuity Issues: ", continuity_issues)

        for continuity_issue in continuity_issues:
            # print("\nContinuity Issue (In Saving RAG Database): ", continuity_issue.get("continuity_issue_id", ""))
            # print("\nScript Id (In Saving RAG Database): ", continuity_issue.get("script_id", ""))
            ids.append(continuity_issue.get("continuity_issue_id", ""))
            # documents.append(continuity_issue.get("script_id", ""))
            documents.append(self.BuildContinuityScenePairDocument(continuity_issue))

            #print("\nRAG Collection (Script Id): ", continuity_issue.get("scene", {}).get("script_id", ""))
            #print("\nRAG Collection (Script Id): ", continuity_issue["scene"]["script_id"])

            metadatas.append({
                "scene_script_id": continuity_issue.get("scene", {}).get("script_id", ""),
                "scene_scene_id": continuity_issue.get("scene", {}).get("scene_id", ""),
                "scene_scene_index": continuity_issue.get("scene", {}).get("scene_index", -1),
                #"scene_slugline": continuity_issue.get("scene", {}).get("slugline", ""),
                "scene_location": continuity_issue.get("scene", {}).get("location", ""),
                "scene_time_of_day": continuity_issue.get("scene", {}).get("time_of_day", ""),
                #"scene_scene_text": continuity_issue.get("scene", {}).get("scene_text", ""),
                #"scene_characters": continuity_issue.get("scene", {}).get("characters", ""),
                #"scene_props": continuity_issue.get("scene", {}).get("props", ""),
                #"scene_actions": continuity_issue.get("scene", {}).get("actions", ""),
                #"scene_normalized_props": continuity_issue.get("scene", {}).get("normalized_props", ""),
                #"scene_wardrobe": continuity_issue.get("scene", {}).get("wardrobe", ""),
                #"scene_action_summary": continuity_issue.get("scene", {}).get("action_summary", ""),

                "related_scene_script_id": continuity_issue.get("related_scene", {}).get("script_id", ""),
                "related_scene_id": continuity_issue.get("related_scene", {}).get("scene_id", ""),
                "related_scene_index": continuity_issue.get("related_scene", {}).get("scene_index", -1),
                #"related_scene_slugline": continuity_issue.get("related_scene", {}).get("slugline", ""),
                "related_scene_location": continuity_issue.get("related_scene", {}).get("location", ""),
                "related_scene_time_of_day": continuity_issue.get("related_scene", {}).get("time_of_day", ""),
                #"related_scene_text": continuity_issue.get("related_scene", {}).get("scene_text", ""),
                #"related_scene_characters": continuity_issue.get("related_scene", {}).get("characters", ""),
                #"related_scene_props": continuity_issue.get("related_scene", {}).get("props", ""),
                #"related_scene_actions": continuity_issue.get("related_scene", {}).get("actions", ""),
                #"related_scene_normalized_props": continuity_issue.get("related_scene", {}).get("normalized_props", ""),
                #"related_scene_wardrobe": continuity_issue.get("related_scene", {}).get("wardrobe", ""),
                #"related_scene_action_summary": continuity_issue.get("related_scene", {}).get("action_summary", ""),                
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        #print("\nRAG Collection: ", self.collection.get())
        return self.collection



    #def GetSceneFromCollectionElement(self, scene_id, collection):
        # 05-22-2026
        # begin here
        
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

        '''
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
        '''
        return prompt.strip()

    def BuildContinuityPromptRAG_ReviewRule(self, current_scene, previous_scene, continuity_issue_id):
        # current_scene_text = self.BuildSceneDocumentText(current_scene)
        # previous_scene_text = self.BuildSceneDocumentText(previous_scene)

        #print("=" * 150)
        #print("\nCurrent Scene: ", current_scene)
        #print("\nPrevious Scene: ", previous_scene)

        #print("\nCurrent Scene Text: ", current_scene)
        current_scene_text = self.BuildSceneDocumentForContinuityIssue(current_scene)

        #print("\nPrevious Scene Text: ", previous_scene)
        previous_scene_text = self.BuildSceneDocumentForContinuityIssue(previous_scene)
        
        #previous_scene_text = f"""
        #                        Previous Scene ID: {previous_scene["scene_id"]}
        #                        Scene Index: {previous_scene["metadatas"]["scene_index"]}
        #                        Similarity Distance: {previous_scene["distance"]}
        #
        #                        {previous_scene["documents"]}
        #                        """     

        print("\nContinuity Issue Id (In Build Continuity Prompt Method): ", continuity_issue_id)
        
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
        {previous_scene_text}

        Return ONLY RFC8259 compliant JSON.

        Important rules:
        - Only report an error if there is clear evidence
        - Do not guess.
        - Compare the current scene against previous scenes.
        - Return JSON only.
        - You are a film script continuity verifier.
        - Never invent props, wardrobe, locations, or actions.
        - Evidence must be directly quoted from scene_text.
        - If the object does not appear in the provided scenes, it does not exist.
        - If uncertain, return has_error=false.
        - Use double quotes for every JSON key.
        - Use double quotes for every string value.
        - Do not use Python dictionary format.
        - Do not use single quotes.
        - Do not include markdown.
        - Do not include explanation.
        - Do not include ```json.
        - The first character must be {{.
        - The last character must be }}.

        IMPORTANT: The "continuity_issue_id" in your response MUST be exactly: {continuity_issue_id}
        Do not modify, invent, or normalize any ID fields.

        Return this JSON format:

        {{
            "continuity_issue_id": {continuity_issue_id},
            "script_id": {current_scene["script_id"]},
            "has_continuity_error": true,

            "detectors": {{
                "rule_based": {{
                    "has_error": false,
                    "errors": null
                }},
                "rag_llm": {{
                    "has_error": true,
                    "errors": {{
                        "scene_id": {current_scene["scene_id"]},
                        "scene_index": {current_scene["scene_index"]},
                        "related_scene_id": {previous_scene["scene_id"]},
                        "issue_type": Continuity Error Type here,
                        "severity": print the severity level here,
                        "description": print description here
                        "evidence": print evidence here
                        "confidence": print confidence level here. it must be number rounded two decimal place
                    }}
                }},
                "fine_tuned_llm": {{
                    "has_error": false,
                    "errors": null
                }},
            }},

            "final_merged_result": {{
                "has_error": true,
                "issue_type": Continuity Error Type here,
                "severity": print the severity level here,
                "description": print description here,
                "evidence": print evidence here,
                "confidence": print confidence level here. it must be number rounded two decimal place,
                "supporting_detectors": [
                    "rag_llm_detector"
                ]
            }}
        }}        
        """                       
        #}}_get_pipeline
        
        return prompt

    def SendPromptToModel(self, prompt):
        model_name = self.model_name

        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                 # <-- this is the setting
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
                #llm_int8_enable_fp32_cpu_offload=True
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

        # To be uncommented
        ###############################################################################################
        #model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                             quantization_config=bnb_config,
        #                                             device_map="auto")
        #                                             torch_dtype=torch.float16)
        ###############################################################################################
        # model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                             quantization_config=bnb_config,
        #                                             device_map="cuda:0",
        #                                             attn_implementation="sdpa")
                                                     #device_map="auto")
                                                     #max_memory={0: "10GiB", "cpu": "30GiB"})
                                                     #torch_dtype=torch.float16)

        # import time
        # inputs_test = tokenizer("Test prompt for speed", return_tensors="pt").to("cuda:0")
        # start = time.time()
        # with torch.no_grad():
        #    outputs = model.generate(**inputs_test, max_new_tokens=50, do_sample=False)
        # elapsed = time.time() - start
        # print(f"\nTokens/sec: {(outputs.shape[1] - inputs_test['input_ids'].shape[1]) / elapsed:.1f}", "\n")

        #import time
        #inputs = tokenizer("Test prompt for speed", return_tensors="pt").to("cuda:0")
        #start = time.time()
        #with torch.no_grad():
        #    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        #elapsed = time.time() - start
        #tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        #print(f"Time:       {elapsed:.2f}s")
        #print(f"Tokens/sec: {tokens_generated / elapsed:.1f}")


        

        
        #print(f"Model name:  {self.model_name}")
        #print(f"Allocated:   {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        #print(f"Reserved:    {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        #cpu_count = sum(1 for _, p in model.named_parameters() if p.device.type == "cpu")
        #gpu_count = sum(1 for _, p in model.named_parameters() if p.device.type != "cpu")
        #total = cpu_count + gpu_count
        #print(f"GPU layers:  {gpu_count}/{total} ({gpu_count/total*100:.1f}%)")
        #print(f"CPU layers:  {cpu_count}/{total} ({cpu_count/total*100:.1f}%)")

        
        ## To be uncommented
        ###############################################################################################
        #llm_pipeline = pipeline("text-generation",
        #                        model=model,
        #                        tokenizer=tokenizer,
        #                        #max_new_tokens=800,
        #                        max_new_tokens=300,
        #                        temperature=0.2,
        #                        do_sample=True,
        #                        return_full_text=False)
        ###############################################################################################

        #llm_pipeline = pipeline("text-generation",
        #                        model=model,
        #                        tokenizer=tokenizer,
        #                        return_full_text=False) 

        llm_pipeline = self._get_pipeline()

        #return llm_pipeline
        #output = llm_pipeline(prompt, max_new_tokens=1024, temperature=0.2, do_sample=True)
        output = llm_pipeline(prompt)

        import time
        inputs_test = tokenizer("Test prompt for speed", return_tensors="pt").to("cuda:0")
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs_test, max_new_tokens=50, do_sample=False)
        elapsed = time.time() - start
        print(f"\nTokens/sec: {(outputs.shape[1] - inputs_test['input_ids'].shape[1]) / elapsed:.1f}", "\n")

        return output

    # def GetStronglyRelatedScenes(current_scene: dict, collection, top_k: int = 10, max_distance: float = 0.8):
    def CheckContinuityWithLLM(self, current_scene, collection):
        related_scenes = self.GetStronglyRelatedScenes(current_scene=current_scene,
                                                       collection=collection,
                                                       top_k=3,
                                                       #top_k=10,
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
        
    def CheckContinuityWithLLM_Review(self, current_scene, prev_scene, continuity_issue_id):
        #def BuildContinuityPromptRAGForRule(self, current_scene, previous_scene, continuity_issue_id):

        #print("Current scene in CheckContinuityWithLLM_Review: ", current_scene)
        print("\nContinuity Issue Id: ", continuity_issue_id, "\n")
        print("\nCurrent Scene Id: ", current_scene["scene_id"], "\n")
        print("\nPrevious Scene Id: ", prev_scene["scene_id"], "\n")
        prompt = self.BuildContinuityPromptRAG_ReviewRule(current_scene=current_scene, previous_scene=prev_scene, continuity_issue_id=continuity_issue_id)

        mistral_prompt = f"<s>[INST] {prompt} [/INST]"

        print("The prompt: ", mistral_prompt, "\n")
        output = self.SendPromptToModel(mistral_prompt)

        # After each inference
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        return output

    def ClearCollection(self):
        results = self.collection.get()

        if (len(results["ids"])) > 0:
            self.collection.delete(ids=results["ids"])

        print("Collection cleared")

    def ClearCollectionContent(self, collection):
        results = collection.get()

        if (len(results["ids"])) > 0:
            collection.delete(ids=results["ids"])

        print("Collection given cleared")