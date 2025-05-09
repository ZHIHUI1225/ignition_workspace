import py_trees
import py_trees_ros
import py_trees_msgs.msg as py_trees_msgs
import rospy
def create_root():
    #behaviours
    root= py_trees.composites.Selector("Selector") #nodo radice nodo selettore
    is_position_object_ok =py_trees.blackboard.CheckBlackboardVariable(
        name="is object in new target pose",
        variable_name="position",
        expected_value="True"
    )# controllo se l'oggetto è in posizione target
    go_to_targetpose= py_trees.composites.Sequence("Sequence") #porto l'oggetto alla posizione target
    check_pushing_plan= py_trees.composites.Selector("Selector")
    is_pushingplan_ok= py_trees.blackboard.CheckBlackboardVariable(
         name="is there a pushing plan",
         variable_name="plan existence",#controllo se il piano di spinta esiste
         expected_value= "True",
    )   
    compute_pushing_plan= py_trees_ros.actions.ActionClient(
         name="compute a new pushing plan",
         action_namespace="/compute",#calcolo nuovo piano di spinta
         action_spec=py_trees.msgs.ComputeAction,
         action_goal=py_trees.msgs.ComputeGoal(),
         override_feedback_message_on_running="computing",
    )
    planner_result=py_trees_ros.subscribers.EventToBlackboard
    execute_pushingplan= py_trees.composites.Sequence("Sequence") #eseguo il piano di pushing
    robot_check_position = py_trees.composites.Selector("Selector") #controllo la posizione del robot  nei confronti dell'oggetto
    is_robot_position_ok = py_trees.blackboard.CheckBlackboardVariable(
         name="is robot in correct object's side",
         variable_name= "robotposition",
         expected_value= "True"
    )
    move_robot= py_trees.composites.Sequence("Sequence") #porto il robot nella giusta posizione nel caso non lo sia
    check_robot_distance= py_trees.composites.Selector("Selector") #mi chiedo se il robot è distante dall'oggetto
    is_robot_distant = py_trees.blackboard.CheckBlackboardVariable(
          name= "distance between position of robot and object>=safety distance",
          variable_name= " robotdistance",
          expected_value= "True"
    )
    detach= py_trees_ros.actions.ActionClient(
            name="detach from object",
            action_namespace="/detach",
            action_spec= py_trees.msgs.DetachAction,
            action_goal=py_trees.msgs.DetachGoal(),
            override_feedback_message_on_running="detaching",
    )
    moveto_approach=py_trees_ros.actions.ActionClient(
            name="move to approach",
            action_namespace="/move",     #mi metto nella giusta posiz.
            action_spec=py_trees.msgs.MoveAction,
            action_goal=py_trees.msgs.MoveGoal(),
            override_feedback_message_on_running="moving",
    )
    approach= py_trees.ros.actions.ActionClient(
            name="approach",
            action_namespace="/approach", # mi avvicino all'oggetto
            action_spec=py_trees.msgs.ApproachAction,
            action_goal=py_trees.msgs.ApproachGoal(),
            override_feedback_message_on_running="approaching",
    )    
    execute_pushing= py_trees.composites.Sequence("Sequence")
        
    select_pushingtrajectory = py_trees.actions.ActionClient(
            name= "select",  #seleziona la traiettoria di pushing
            action_namespace="/select",
            action_spec=py_trees.msgs.SelectAction,
            action_goal=py_trees.msgs.SelectGoal(),
            override_feedback_message_on_running="selecting",
    )
    execute_pushingtrajectory=py_trees.actions.ActionClient(
            name= "execute",    #esegui la traiettoria di pushing
            action_namespace="/execute",
            action_spec=py_trees.msgs.ExecuteAction,
            action_goal=py_trees.msgs.ExecuteGoal(),
            override_feedback_message_on_running="executing",
    )
    #struttura albero
    root.addchildren([is_position_object_ok,go_to_targetpose])
    go_to_targetpose.addchildren([check_pushing_plan,execute_pushingplan])
    check_pushing_plan.addchildren([is_pushingplan_ok,compute_pushing_plan,
                                    planner_result])
    execute_pushingplan.addchildren([robot_check_pushing,execute_pushing])
    robot_check_position.addchildren([is_robot_position_ok,move_robot])
    move_robot.addchildren([check_robot_distance,moveto_approach,approach])
    check_robot_distance.addchildren([is_robot_distance,detach])
    execute_pushing.addchildren([select_pushingtrajectory,
                                 execute_pushongtrajectory])
                               
    return root 
     
def main(): #codice che python esegue
    rospy.init_node("tree")     #inizializzo
    root=create_root()     #chiamo la funzione create root
    behaviour_tree=py_trees_ros.trees.BehaviourTree(root)
                                           
       

