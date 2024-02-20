
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    db
)


def connect_milvus(host, port):
    conn = connections.connect(host=host, port=port)
    return conn


def create_database(database_name):
    database = db.create_database(database_name)
    return database


def create_collection(collection_name):
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name='document', dtype=DataType.JSON),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=768)
    ]
    schema = CollectionSchema(fields, " fields include id and document, document is json format")
    the_collection = Collection(name=collection_name, schema=schema)
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    the_collection.create_index('embedding', index)

    return the_collection


def insert_to_collection(entity, collection_):
    id_, document_, embedding_ = entity[0], entity[1], entity[2]
    to_insert = [[id_], [document_], [embedding_]]
    insert_result = collection_.insert(to_insert)
    return insert_result


def flush_collection(collection_):
    collection_.flush()


def load_collection(ip, port, collection_name):
    connect_milvus(ip, port)
    db.using_database('default')
    collection_ = Collection(collection_name)
    state = utility.load_state(collection_name)
    if state == 2:
        utility.wait_for_loading_complete(collection_name)
        return collection_
    if state == 3:
        print("collection is already loaded!")
        return collection_
    if state == 1:
        print("collection is not loaded! Load it now!")
        collection_.load()
        utility.wait_for_loading_complete(collection_name)
        if utility.load_state(collection_name) == 3:
            print("collection is loaded!")
        return collection_


def search_data(collection, vector_data, result_num=2, threshold=0.75):
    # connect_milvus(ip, port)
    # db.using_database('default')
    # collection_ = Collection(collection_name)
    # collection_.release()
    # while True:
    #     try:
    #         collection_.load(timeout=60)
    #         utility.wait_for_loading_complete(collection_name)
    #         break
    #     except pymilvus.exceptions.MilvusException:
    #         continue

    search_params = {
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }

    state = utility.load_state(collection.name)
    if state == 3:
        pass
    if state == 2:
        utility.wait_for_loading_complete(collection.name)
    if state == 1:
        print("collection is not loaded! Load it now!")
        collection.load()
        utility.wait_for_loading_complete(collection.name)
        if utility.load_state(collection.name) == 3:
            print("collection is loaded!")

    search_result = collection.search(vector_data,
                                      anns_field='embedding',
                                      param=search_params,
                                      limit=result_num,
                                      expr=None,
                                      output_fields=['document'])
    # collection.release()
    search_result_final = [search_result[0][i].entity.get('document') for i in range(result_num)]
    search_result_final_score = [search_result[0][i].entity.distance for i in range(result_num)]
    threshold_result_final = [item for item in search_result_final if
                              search_result_final_score[search_result_final.index(item)] >= threshold]
    scores_final = [item for item in search_result_final_score if item >= threshold]
    return search_result_final, threshold_result_final, scores_final


def release_collection(collection):
    collection.release()
    print("collection released!")


def drop_collection(collection_name):
    utility.drop_collection(collection_name)


