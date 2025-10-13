from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('',views.home,name="home"),
    path('api/projects/create/', views.create_project, name='create_project'),
    path('api/projects/', views.list_projects, name='list_projects'),
    path('api/projects/<int:pk>/', views.get_project, name='get_project'),
    path('api/projects/<int:pk>/update/', views.update_project, name='update_project'),
    path('api/projects/<int:pk>/delete/', views.delete_project, name='delete_project'),

    #FILE
    path('api/files/create/', views.create_file, name='create_file'),
    path('api/files/', views.list_files, name='list_files'),
    path('api/files/<int:pk>/', views.get_file, name='get_file'),
    path('api/files/<int:pk>/update/', views.update_file, name='update_file'),
    path('api/files/<int:pk>/delete/', views.delete_file, name='delete_file'),
    path('api/files/extract_first_column/', views.extract_first_column, name='extract_first_column'),



    path('api/projects/<int:project_id>/files/', views.files_by_project, name='files_by_project'),
    path('api/files/<int:file_id>/tables/', views.get_table_names_by_file, name='get_table_names_by_file'),
    path('api/files/<int:file_id>/prompt_tables/', views.process_select_query, name='process_select_query'),
    path('api/files/<int:file_id>/create_error_records/', views.create_error_records, name='create_error_records'),
    path('api/files/<int:file_id>/get_error_table/', views.get_error_table, name='get_error_table'),

    path('api/Get_Post_Load_Report/<int:pre_id>/<int:post_id>/', views.Get_Post_Load_Report, name='Get_Post_Load_Report'),
    path('api/Get_Post_Load_Pivot/<int:pre_id>/<int:post_id>/', views.Get_Post_Load_Pivot, name='Get_Post_Load_Pivot'),
    path('api/get_tables_by_project/<int:pid>/', views.get_tables_by_project, name='get_tables_by_project'),





















#     path('api/sapconn/',views.SAPconn,name="SAPconn"),
#      path('api/saptables/<int:load>/<int:connection_id>/',views.SAPtables,name="SAPtables"),
#     path('api/hanaconn/',views.HANAconn,name="HANAconn"),
#     path('api/hanatables/<int:p_id>/<str:c_name>/',views.HANAtables,name="hanatables"),
#     path('api/hanadata/',views.HANAtables,name="HANAtables"),


#     # path('api/saptables_to_sqlite/<int:connection_id>/',views.saptables_to_sqlite,name="sqltolite"),
#     path('api/SAPTableSearch/<str:tab>/<int:connection_id>/',views.SAPTableSearch,name="SAPTableSearch"),

#     # project CURD
#     path('api/Pcreate/',views.ProjectCreate,name="Pcreate"),
#     path('api/Pget/',views.ProjectGet,name="Pget"),
#     path('api/PgetSingle/<int:p_id>/',views.ProjectGetSingle,name="PgetSingle"),
#     path('api/PUpdate/<int:pk>/',views.projectUpdate,name="PUpdate"),
#     path('api/PDelete/<int:pk>/',views.project_delete,name="PDelete"),

    
    
#     # connection CURD
#     path('api/Ccreate/',views.ConnectionCreate,name="Ccreate"),
#     path('api/Cupdate/<int:p_id>/<str:c_name>/', views.ConnectionUpdate, name='Cupdate'),
#     path('api/Cget/',views.ConnectionGet,name="Cget"),
#     path('api/Cdelete/<int:p_id>/<str:c_name>/',views.connectionDelete,name="Cdelete"),
#     path('api/CgetSingle/<int:p_id>/<str:c_name>/',views.ConnectionGetSingle,name="CgetSingle"),
#     path('api/Crename/<str:re_val>/<int:p_id>/<str:c_name>/',views.connectionRename,name="Crename"),

    # path('xls/',views.xls_read,name="xls_read"),
#     path('api/ObjGet/<int:oid>/',views.objects_get,name="objects_get"),
#     path('api/ObjCreate/',views.objects_create,name="objects_create"),
#     path('api/ObjUpdate/<int:oid>/',views.objects_update,name="objects_update"),
#     path('api/ObjDelete/<int:oid>/',views.objects_delete,name="objects_delete"),
 
 
#     #Rules Page API's
#     path('api/PdataObject/<int:pid>/<str:ptype>/',views.project_dataObject,name="project_dataObject"),
#     path('api/Osegements/<int:pid>/<int:oid>/',views.DataObject_Segements,name="DataObject_Segements"),
#     path('api/Sfields/<int:pid>/<int:oid>/<int:sid>/',views.Segements_Fields,name="Segements_Fields"),
#     path('api/getTable/<int:sid>/',views.getTableData,name="getTableData"),
#     path('api/execute_queries/<int:pid>/<int:oid>/<int:sid>/',views.execute_queries,name="execute_queries"),
#     path('api/execute_selection_criteria/<int:pid>/<int:oid>/<int:sid>/',views.execute_selection_criteria,name="execute_selection_criteria"),
#     path('api/getLatestVersion/<int:pid>/<int:oid>/<int:sid>/',views.getLatestVersion,name="getLatestVersion"),
#     path('api/applyOneToOne/<int:pid>/<int:oid>/<int:sid>/',views.applyOneToOne,name="applyOneToOne"),
#     path('api/getSapTableData',views.getSapTableData,name="getSapTableData"),
#     # need to check this url in frontend where we are using
#     path('api/upload_Bussiness_rules',views.upload_Bussiness_rules,name="upload_Bussiness_rules"),
#     path('api/get_error_table/<int:pid>/<int:oid>/<int:sid>/',views.get_error_table,name="get_error_table"),
    

#  #CHAT
#     path('api/createChat/',views.CreateChat,name="createChat"),
#     path('api/getChat/<int:pid>/<int:oid>/<int:sid>/',views.getChat,name="getChat"),
#     path('xls/',views.xls_read,name="xls_read"),
#     path('tableDelete/',views.tableDelete,name="tableDelete"),
#     path('delete_table_data/',views.delete_table_data,name="delete_table_data"),
    

#   # file CURD
#     path('api/fcreate/',views.fileCreate,name="fcreate"),
#     path('api/fupdate/<int:p_id>/<str:f_name>/', views.fileUpdate, name='fupdate'),
#     path('api/fget/',views.fileGet,name="fget"),
#     path('api/fdelete/<int:p_id>/<str:f_name>/',views.fileDelete,name="fdelete"),
#     path('api/fgetSingle/<int:p_id>/<str:f_name>/',views.fileGetSingle,name="fgetSingle"),
#     path('api/frename/<str:re_val>/<int:p_id>/<str:f_name>/',views.fileRename,name="frename"),
 
#     #file
    path('excel/', views.GetXL.as_view(), name = "by"),
#     path('txt/', views.GetTXT.as_view(), name = "hi"),
#     path('csv/', views.GetFile.as_view(), name = "by"),
    # path('xlsx/', views.GetXLSheet.as_view(), name = "hlo"),


#       #VersionsRule API's
#     path('api/VersionRuleCreate/',views.VersionRuleCreate,name="VersionRuleCreate"),
 
 
 
#     #Save Rule Api's
#     path('api/CreateSaveRules/',views.SaveRuleCreate,name="CreateSaveRules"),
#     path('api/GetSaveRules/<int:pid>/<int:oid>/<int:sid>/',views.GetSaveRule,name="GetSaveRule"),
 
#     # path('api/RuleVersions/<int:pid>/<int:oid>/<int:sid>/',views.RuleVersions,name="RuleVersions"),
#     path('api/VersionData/<int:pid>/<int:oid>/<int:sid>/<int:vid>/',views.VerisonData,name="VerisonData"),


#      path('api/getSapTableData',views.getSapTableData,name="getSapTableData"),

#     #success Factors 
#     path('saveSuccessFactors/', views.saveSuccessFactors, name = "saveSuccessFactors"),
#     path('api/getSfTableData/<int:oid>/', views.getSfTableData, name = "getSfTableData"),
#     path('reUploadSuccessFactors/<int:oid>/', views.reUploadSuccessFactors, name = "reUploadSuccessFactors"),



#     #encryption
#     # path('demo_encryption/', views.demo_encryption, name = "demo_encryption"),


#     #validations
#     path('api/create_Validation_Table/', views.create_Validation_Table, name = "create_Validation_Table"),
#     path('api/Insert_Data_Into_ValidationTable/', views.Insert_Data_Into_ValidationTable, name = "Insert_Data_Into_ValidationTable"),
#     path('api/create_PreLoad_Tables/', views.create_PreLoad_Tables, name = "create_PreLoad_Tables"),
#     path('api/validate_mandatory_fields/', views.validate_mandatory_fields, name = "validate_mandatory_fields"),
#     # path('api/validate_Lookup_fields/', views.validate_Lookup_fields, name = "validate_Lookup_fields"),


#     # #Final Report
#     # path('api/final_report/<int:project_id>/', views.final_report, name = "final_report"),
#     # path('api/get_report_table/<int:segment_id>/<str:table_type>/', views.get_report_table, name = "get_report_table"),
#     # path('api/download_final_report/<int:segment_id>/<str:table_type>/', views.download_final_report, name = "download_final_report"),
#     # path('api/get_numberOfLookupValidationFailed_Count/', views.get_numberOfLookupValidationFailed_Count, name = "get_numberOfLookupValidationFailed_Count"),




#     path('api/sqltable_to_excel/', views.sqltable_to_excel, name = "sqltable_to_excel"),
#     path('api/excel_data_to_sqllite/', views.excel_data_to_sqllite, name = "excel_data_to_sqllite"),
    
#     path('plot/<str:pid>/<str:oid>/<str:sid>/', views.GetPlot.as_view(), name = "plot"),
#     path('particular/<str:pid>/<str:oid>/<str:sid>/<str:fname>/', views.GetExactGraph.as_view(), name = "Exact"),


#     path('api/download_database/', views.download_database, name = "download_database"),
#     path('api/download_project_data/<int:project_id>/', views.download_project_data, name = "download_project_data"),
#     path('api/import_project_data/', views.import_project_data, name = "import_project_data"),
#     path('api/delete_table_data/', views.delete_table_data, name = "delete_table_data"),


#     #PRELOAD_TABLE

#     path('api/get_preLoad_table/<int:pid>/<int:oid>/<int:sid>/', views.get_preLoad_table, name = "get_preLoad_table"),
#     path('api/get_validation_table/<int:pid>/<int:oid>/<int:sid>/', views.get_validation_table, name = "get_validation_table"),
#     path('api/check_is_initial_version/<int:pid>/<int:oid>/<int:sid>/', views.check_is_initial_version, name = "check_is_initial_version"),
#     path('api/file_upload_bussiness/',views.file_upload_bussiness,name="file_upload_bussiness")
]
